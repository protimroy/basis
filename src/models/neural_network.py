import torch;
from ..layers.input import InputLayer
from ..layers.layer import Layer
from ..layers.dense import DenseLayer
from ..layers.weight import WeightLayer
from ..losses.cross_entropy import Loss
from ..optimizers.gradient_descent import GD
from ..utils.tensorboard_logs import create_tensorboard_log, log_scalar, log_histogram, close_tensorboard_log
from ..utils.gpu_monitor import GPUMonitor

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import time


class NeuralNetwork:
    """
    This class respresents a Neural Networks.
    """
    def __init__( self, *layers, device='cpu' ):
        """
        Constructor for the Neural Network class.
        
        Args:
            layers                    : ( list ) List of layers in the network.
            device                    : device to use for computation

        Returns:
            None

        Attributes:
            self.hyperparams          : ( dict ) hyperparameters for the network.
            self.layers               : ( list ) list of layers in the network.
            self.layer_name           : ( list ) list of names of the layers in the network.
            self.input_layer          : ( object ) input layer of the network.
            self.output_layer         : ( object ) output layer of the network.
            self.weights              : ( list ) list of weights in the network.
            self.weight_names         : ( list ) list of names of the weights in the network.
            self.loss_fn              : ( object ) loss function for the network.
            self.optimizer            : ( object ) optimizer for the network.
            self.learnable_parameters : ( dict ) dictionary of learnable parameters in the network.
            self.device               : device to use for computation

        Raises:
            None 
        """
        self.layers = list( layers )
        self.device = device
        
        if not self.layers or not isinstance( self.layers[0], InputLayer ):
            raise ValueError( f"First layer must be an InputLayer, not {type(self.layers[0]).__name__}" );
        
        # hyperparameters dictionary
        self.hyperparams = self.layers[0].hyperparams;

        # layers
        self.input_layer = self.layers[0];
        self.output_layer = self.layers[-1];

        self.weights = []
        for i in range( len( self.layers) - 1 ):
            src_layer = self.layers[i];
            dest_layer = self.layers[i + 1];

            if isinstance( src_layer, ( InputLayer, DenseLayer ) ) and isinstance( dest_layer, DenseLayer ):
                self.weights.append( WeightLayer(src_layer, dest_layer, device=self.device) );
            else:
                raise ValueError( f"Unsupported layer connection between {type(src_layer).__name__} and {type(dest_layer).__name__} for MLP." );

        self.loss_fn = None;
        self.optimizer = None;

        self.trainable_parameters = self._collect_parameters();
    
    def _collect_parameters( self ):
        """
        Collects all trainable parameters from the layers and weights in the network.
        """
        trainable_parameters = {};
        for layer in self.layers:
            if isinstance( layer, DenseLayer ):
                trainable_parameters.update( layer.get_parameters() );
        
        for weight_layer in self.weights:
            trainable_parameters.update( weight_layer.get_parameters() );

        return trainable_parameters;

    def clear_all_gradients(self):
        """Clears .grad attribute for all trainable parameters."""
        for param_tensor in self.trainable_parameters.values():
            if param_tensor.grad is not None:
                param_tensor.grad.zero_()
   
    def __rshift__( self, other ):
        """
        Overwrite the right shift operator to add a layer to the network
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        if isinstance( other, Layer ) or isinstance( other, InputLayer ):
            # add weights between the layers, append the weight name to the list
            self.weights.append( WeightLayer( self.layers[-1], other, device=self.device ) );
            
            # add the layer to the network, set the output layer to the last layer, and append the layer name to the list
            self.layers.append( other );
            self.output_layer = self.layers[-1];

            # add the weights and biases to the trainable parameters
            self.trainable_parameters[self.weights[-1].name] = self.weights[-1].weights;
            self.trainable_parameters[other.name] = other.bias;

            return self;
        else:
            raise TypeError( f"Invalid layer type: {type(other).__name__}. Expected Layer or InputLayer." );

    def output( self, inputs ):
        """
        Calculate the output of the network.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        return self.output_layer.output( inputs );

    def compile(self, loss: Loss, optimizer_class, learning_rate):
        self.loss_fn = loss
        #self.optimizer = optimizer_class(params=self.trainable_parameters, lr=learning_rate)
        self.optimizer = optimizer_class(params=list(self.trainable_parameters.values()), lr=learning_rate)


    def forward( self, inputs: torch.Tensor ) -> dict:
        """
        Executes the forward pass through the entire network.

        Args:
        Returns:
        Attributes:
        Raises:
        """
        fwd_output = {};
        current_output = self.input_layer.output( inputs );
        fwd_output[self.input_layer.name] = current_output;

        for i in range( len( self.weights ) ):
            weight_layer = self.weights[i];
            weighted_input = weight_layer.output( current_output );
            fwd_output[weight_layer.name] = weighted_input;

            layer = self.layers[i + 1];
            current_output = layer.output( weighted_input );
            fwd_output[layer.name] = current_output;

        fwd_output['output'] = current_output;
        return fwd_output;

    def backward(self, grad_loss_wrt_output: torch.Tensor):
        # Start with the gradient of the loss with respect to the final output of the network (dL/da_output)
        current_grad = grad_loss_wrt_output

        # Backward pass through the output DenseLayer (dL/da_output -> dL/dz_output -> dL/dX_output_from_weights)
        # The backward method of DenseLayer takes dL/da and returns dL/dZ.
        grad_wrt_weighted_output = self.layers[-1].backward(current_grad)

        # Iterate backwards through weight connections and dense layers
        # self.weights[i] connects self.layers[i] to self.layers[i+1]
        for i in range(len(self.weights) - 1, -1, -1):
            # The grad_wrt_weighted_output is dL/dZ for the current DenseLayer.
            # This needs to be passed to the WeightLayer's backward method.
            weight_conn = self.weights[i]

            # WeightLayer.backward takes dL/dZ (grad_output_wrt_Z) and returns dL/dX_previous (grad w.r.t input of weight layer)
            # This dL/dX_previous is then passed to the previous DenseLayer (or InputLayer).
            grad_wrt_previous_layer_output = weight_conn.backward(grad_wrt_weighted_output)

            # If there's a preceding DenseLayer, propagate back through its activation
            if i > 0: # Not the input layer
                dense_layer_before_weights = self.layers[i]
                # The backward method of DenseLayer expects dL/da (grad_output), so we pass dL/dX_previous from WeightLayer.
                grad_wrt_weighted_output = dense_layer_before_weights.backward(grad_wrt_previous_layer_output)
            else: # This is the input layer, it doesn't have trainable parameters
                pass # InputLayer.backward just returns its input grad, which is fine
    
    def reset( self ):
        """
        Resets the network to its initial state.
        
        Args:
        Returns:
        Attributes:
        Raises:
        """
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.reset()
        
        for weight_layer in self.weights:
            weight_layer.reset()

        # Clear gradients
        self.clear_all_gradients()
    
    
    def train(self, train_data, train_labels, epochs, 
          save_param_history=False, param_save_interval=100,
          profile_gpu=True):
        """
        Enhanced training method with GPU profiling and monitoring.
        """
        if self.loss_fn is None or self.optimizer is None:
            raise RuntimeError("Network must be compiled with a loss function and optimizer before training.")
        
        # Check device
        device = next(iter(self.trainable_parameters.values())).device  # Changed: Added iter()
        gpu_monitor = GPUMonitor(device.type)
        
        num_samples = train_data.shape[0]
        
        # Initialize parameter history tracking if requested
        if save_param_history:
            self.param_history = []
            self.loss_history = []
            
            def save_param_snapshot():
                snapshot = {}
                for name, param in self.trainable_parameters.items():
                    snapshot[name] = param.data.clone()
                return snapshot
            
            self.param_history.append(save_param_snapshot())
        
        # Enhanced logging with GPU metrics
        log_dir = f"runs/nn_training_gpu_{device.type}"
        experiment_metadata = {
            "optimizer": type(self.optimizer).__name__,
            "loss_fn": type(self.loss_fn).__name__,
            "epochs": epochs,
            "device": str(device),
            "gpu_name": gpu_monitor.gpu_name if gpu_monitor.gpu_available else "N/A"
        }
        
        self.writer, self.run_id = create_tensorboard_log(
            log_dir,
            hyperparams=self.hyperparams,
            extra_metadata=experiment_metadata,
        )
        
        # Log initial GPU state
        gpu_monitor.log_to_tensorboard(self.writer, 0)
        
        # Initialize profiler for detailed GPU analysis
        if profile_gpu and device.type == 'cuda':
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=10, warmup=10, active=20, repeat=2),
                on_trace_ready=tensorboard_trace_handler(f'{log_dir}/profiler'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()
        else:
            prof = None
        
        # Training loop with GPU monitoring
        training_start_time = time.time()
        batch_times = []
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            # Clear gradients
            self.clear_all_gradients()
            
            # Forward and backward passes
            for i in range(num_samples):
                input_sample = train_data[i].unsqueeze(1)
                target_sample = train_labels[i].unsqueeze(1)
                
                # Time individual operations
                if epoch % 100 == 0:  # Sample timing occasionally
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    fwd_start = time.time()
                
                # Forward pass
                output = self.forward(input_sample)
                loss_value = self.loss_fn(output['output'], target_sample)
                
                if epoch % 100 == 0:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    fwd_time = time.time() - fwd_start
                    self.writer.add_scalar('Timing/forward_pass_ms', fwd_time * 1000, epoch * num_samples + i)
                
                epoch_loss += loss_value.item()
                
                # Backward pass
                if epoch % 100 == 0:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    bwd_start = time.time()
                
                grad_loss_wrt_net_output = self.loss_fn.backward()
                self.backward(grad_loss_wrt_net_output)
                
                if epoch % 100 == 0:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    bwd_time = time.time() - bwd_start
                    self.writer.add_scalar('Timing/backward_pass_ms', bwd_time * 1000, epoch * num_samples + i)
            
            # Update parameters
            self.optimizer.step()
            
            # Compute epoch metrics
            epoch_loss /= num_samples
            epoch_time = time.time() - epoch_start_time
            batch_times.append(epoch_time)
            
            # Save history
            if save_param_history:
                self.loss_history.append(epoch_loss)
                if (epoch + 1) % param_save_interval == 0 or epoch == epochs - 1:
                    self.param_history.append(save_param_snapshot())
            
            # Log metrics
            self.writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            self.writer.add_scalar('Timing/epoch_time_ms', epoch_time * 1000, epoch)
            self.writer.add_scalar('Timing/samples_per_second', num_samples / epoch_time, epoch)
            
            # Log GPU metrics
            if epoch % 10 == 0:  # Log GPU metrics every 10 epochs
                gpu_monitor.log_to_tensorboard(self.writer, epoch)
                
                # Log GPU memory statistics
                if device.type == 'cuda':
                    self.writer.add_scalar('GPU/memory_cached_mb', 
                                        torch.cuda.memory_reserved() / 1024**2, epoch)
                    self.writer.add_scalar('GPU/memory_allocated_mb', 
                                        torch.cuda.memory_allocated() / 1024**2, epoch)
            
            # Profile step
            if prof:
                prof.step()
            
            # Log detailed info periodically
            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time*1000:.1f}ms")
                if epoch % 500 == 0 and epoch > 0:
                    gpu_monitor.print_current_status()
        
        # Training complete - log final statistics
        total_training_time = time.time() - training_start_time
        avg_epoch_time = sum(batch_times) / len(batch_times)  # Changed from np.mean()
        
        self.writer.add_text('Summary/total_training_time', f"{total_training_time:.2f} seconds")
        self.writer.add_text('Summary/average_epoch_time', f"{avg_epoch_time*1000:.2f} ms")
        self.writer.add_text('Summary/device', str(device))
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            self.writer.add_text('Summary/peak_gpu_memory_mb', f"{peak_memory:.1f}")
            print(f"\nPeak GPU memory usage: {peak_memory:.1f} MB")
        
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Average time per epoch: {avg_epoch_time*1000:.2f} ms")
        
        # Stop profiler
        if prof:
            prof.stop()
            print(f"\nProfiler trace saved to: {log_dir}/profiler")
        
        # Close writer
        self.writer.close()
    