import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import time
import datetime
from torch.utils.data import RandomSampler, TensorDataset, SequentialSampler, random_split
import random

class Net:
    def __init__(self, device):
     
        self.device = device

        self.g = torch.Generator()
        self.g.manual_seed(42)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_train_val_dataloaders(self, input_ids, attention_masks, labels, batch_size):

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size, # Trains with this batch size.
                    num_workers = 0,
                    worker_init_fn = self.seed_worker,
                    generator = self.g
                )
        
        

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = 32, # Evaluate with this batch size.
                    num_workers = 0,
                    worker_init_fn = self.seed_worker,
                    generator = self.g
                )

        return train_dataloader, validation_dataloader

    def train_bert(self, input_ids, attention_masks, labels, LEARNING_RATE, BATCH_SIZE, EPOCHS):

        train_dataloader, validation_dataloader = self.get_train_val_dataloaders(input_ids, attention_masks, labels, BATCH_SIZE)

        # ========================================
        #   Load BertForSequenceClassification
        # ========================================

        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        self.clf = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            return_dict=False
        )

        # Set number of epochs
        epochs = EPOCHS
        learning_rate = LEARNING_RATE
        batch_size = BATCH_SIZE

        # Tell pytorch to run this model on the GPU.
        self.clf.cuda()

        # ========================================
        #               Create Adam Optemizer
        # ========================================

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(self.clf.parameters(),
                            lr = learning_rate, # args.learning_rate - default is 5e-5, 
                            eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

            
        # ========================================
        #               Create Scheduler
        # ========================================


        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)


        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.clf.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                #if step == 0:
                #  print(type(batch))
                #  print(batch[0])

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.clf.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = self.clf(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
            
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.clf.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = self.clf(b_input_ids, 
                                            token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels=b_labels)
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
                

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

            # Save performance of this epoch
            #if epoch_i == epochs:
            #    self.export_training_summary(learning_rate, epoch_i + 1, batch_size, n_round, training_stats)
           #     #save_model(model, learning_rate,  epoch_i + 1, batch_size)

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        return training_stats

    def predict_bert(self, input_ids, attention_masks, labels):

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

        # Prediction on test set
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
        #model.cuda()
        self.clf.eval()

        #loaded_model.cuda()
        #loaded_model.eval()

        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in prediction_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            #batch = tuple(t for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.clf(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        print('    DONE.')
        return flat_predictions

    def predict_prop_bert(self, input_ids, attention_masks, labels):

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

        # Prediction on test set
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
        #model.cuda()
        self.clf.eval()

        #loaded_model.cuda()
        #loaded_model.eval()

        # Tracking variables 
        predictions , true_labels = [], []

        # Predict 
        for batch in prediction_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            #batch = tuple(t for t in batch)
            
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.clf(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        # For each sample, pick the label with the higher score.
        #flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_predictions = torch.from_numpy(flat_predictions)
        print('    DONE.')
        return flat_predictions
        