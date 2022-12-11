import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.activations import softmax, relu, sigmoid

class MatchNet(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.key_layer = Dense(units=config['hidden_size'])
        self.dropout = Dropout(rate=config['dropout_rate'])
    
    def call(self, inputs):
        p, q = inputs
        
        # Compute keys
        keys = self.key_layer(q)
        
        # Calculate and normalize attention of q on p
        attention_weights = tf.matmul(p, keys, transpose_b=True)
        attention_weights = softmax(attention_weights)
        
        # Finally get attentionized output of q
        outputs = relu(tf.matmul(attention_weights, q))
        
        return outputs      
        

class GateLayers(Model):
    """ A gate for fusing two inputs of the same size. Similar to a minimal gated unit: 
    https://en.wikipedia.org/wiki/Gated_recurrent_unit#Minimal_gated_unit

    Attributes:
        x_gate: forget/memorize layer for the first input
        y_gate: forget/memorize layer for the second input
    
    Usage:
        gate_layer((x, y)) -> gated_output
        
        `gated_output` is the same size as x, and y, and is a combination of the two inputs
        based on a remember/forget value computed for each element, where
        
        $f_t = sigmoid(W_x x + W_y y), Output = f_t x + (1 - f_t)y$
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.x_gate = Dense(units=config['hidden_size'])
        self.y_gate = Dense(units=config['hidden_size'])
    
    def call(self, inputs):
        x, y = inputs
        
        gate_value = sigmoid(self.x_gate(x) + self.y_gate(y))
        outputs = gate_value * x + (1 - gate_value) * y
        
        return outputs


class DCMN(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize parameters relevant to task & trainingt
        self.n_choices = config['n_choices']
        self.dropout = config['dropout']
        
        # Initialize layers
        self.bert = config['pre_trained_bert']
        self.classifier = config['classifier']
        self.gate_layer = config['gate_layer'](config)
        self.match_layer = MatchNet(config)
        
        
    def call(self, inputs):
        passage, question, options = inputs
        
        p_embedding = self.bert(passage)
        q_embedding = self.bert(question)
        o_embedding = self.bert(options)
        
        po = tf.math.reduce_max(self.match_layer(p_embedding, o_embedding), axis = 1)
        op = tf.math.reduce_max(self.match_layer(o_embedding, p_embedding), axis = 1)
        qo = tf.math.reduce_max(self.match_layer(q_embedding, o_embedding), axis = 1)
        oq = tf.math.reduce_max(self.match_layer(o_embedding, q_embedding), axis = 1)
        pq = tf.math.reduce_max(self.match_layer(p_embedding, q_embedding), axis = 1)
        qp = tf.math.reduce_max(self.match_layer(q_embedding, p_embedding), axis = 1)
        
        po_fused = self.gate_layer((po, op))
        pq_fused = self.gate_layer((pq, qp))
        qo_fused = self.gate_layer((qo, oq))
        
        all_attention_values = tf.concat((po_fused, pq_fused, qo_fused), axis=1)
        
        answer_logits = tf.reshape(self.classifier(all_attention_values), shape=(-1,self.n_choices))
        
        return answer_logits
    
    def main():
        default_classifier = Dense(1, activation='softmax')
        
        default_config = {'n_choices': 4,
                    'hidden_size': 512,
                    'pre_trained_bert': None,
                    'gate_layer': GateLayers,
                    'classifier': default_classifier}
        
        dcmn_model = DCMN(default_config)
        
    if __name__ == "__main__":
        main()
        
        
        
        
        