import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras.activations import softmax, relu, sigmoid

class MatchNet(Model):
    """Subnetwork for attention between P/Q/O pairs. Returns the attention
    values of the second input from the first, using input directly as query/
    values
    
    Attributes:
        key_layer: layer to compute keys from input p
        dropout: dropout rate, if used in training
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.key_layer = Dense(units=config['hidden_size'])
        self.dropout = Dropout(rate=config['dropout'])
    
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
    """Dual Co-Matching Network Implementation. Calculates two-sided attention of all
    qassage/question/answer embeddings and uses it to predict the correct choice.

    Usage:
        DCMN((p, q, o))) -> tf.tensor([b_size, n_choices]), where `p`, `q` and `o` each is a tensor of strings
        from the dataset. Returns a softmax vector of length n_choices for the prediction.
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize parameters relevant to task & trainingt
        self.n_choices = config['n_choices']
        self.dropout = config['dropout']
        self.hidden_size = config['hidden_size']
        
        # Initialize layers
        self.bert_preprocess = config['bert_preprocess']
        self.bert = config['bert']
        self.match_layer = MatchNet(config)
        self.gate_layer = config['gate_layer'](config)
        self.classifier = config['classifier']
        
        
    def call(self, inputs):
        batch_size = len(inputs)
        passage, question, options = inputs
        
        p_tokens = self.bert_preprocess(passage)
        q_tokens = self.bert_preprocess(question)
        o_tokens = self.bert_preprocess(options)
        #tf.map_fn(lambda option: self.bert_preprocess(option), options)
        
        p_embedding = self.bert(p_tokens)['sequence_output']
        q_embedding = self.bert(q_tokens)['sequence_output']
        o_embedding = self.bert(o_tokens)['sequence_output']
        #tf.map_fn(lambda option: self.bert(option), o_tokens)
        
        po = tf.math.reduce_max(self.match_layer((p_embedding, o_embedding)), axis = -2)
        op = tf.math.reduce_max(self.match_layer((o_embedding, p_embedding)), axis = -2)
        qo = tf.math.reduce_max(self.match_layer((q_embedding, o_embedding)), axis = -2)
        oq = tf.math.reduce_max(self.match_layer((o_embedding, q_embedding)), axis = -2)
        pq = tf.math.reduce_max(self.match_layer((p_embedding, q_embedding)), axis = -2)
        qp = tf.math.reduce_max(self.match_layer((q_embedding, p_embedding)), axis = -2)
        
        po_fused = self.gate_layer((po, op))
        pq_fused = self.gate_layer((pq, qp))
        qo_fused = self.gate_layer((qo, oq))
        
        all_attention_values = tf.concat((po_fused, pq_fused, qo_fused), axis=-1)
        answer_logits = self.classifier(all_attention_values)
        
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