import tensorflow as tf
import numpy as np
import pickle


def create_encoder_network(full_dim, reduced_dim, reduced_basis, input_shift=None, variable_name='m'):
	# First instance input layers
    input_name = '%s_input' %(variable_name)
    layer_name = 'reduced_basis_%s' %(variable_name)

    full_input = tf.keras.layers.Input(shape=(full_dim), name=input_name)

    if input_shift is None:
        reduced_input = tf.keras.layers.Dense(reduced_dim, use_bias=False, name=layer_name)(full_input)
    else:
        input_shift = tf.convert_to_tensor(input_shift, dtype=tf.float32)
        reduced_input = tf.keras.layers.Dense(reduced_dim, use_bias=False, name=layer_name)(full_input - input_shift)

    encoder_network = tf.keras.models.Model(full_input, reduced_input)

    if reduced_basis is not None:
        encoder_network.get_layer(layer_name).set_weights([reduced_basis])
        encoder_network.get_layer(layer_name).trainable = False

    return encoder_network

def create_decoder_network(reduced_dim, full_dim, reduced_basis, output_shift=None, variable_name='u'):
	# First instance input layers
    input_name = '%s_reduced' %(variable_name)
    layer_name = 'reduced_basis_%s' %(variable_name)

    reduced_output = tf.keras.layers.Input(shape = (reduced_dim, ), name=input_name)
    if output_shift is None:
        # print("No shift")
        full_output = tf.keras.layers.Dense(full_dim, use_bias=False, name=layer_name)(reduced_output)
    else:
        # print("With shift")
        full_output = tf.keras.layers.Dense(full_dim, name=layer_name)(reduced_output)

    decoder_network = tf.keras.models.Model(reduced_output, full_output)

    if reduced_basis is not None:
        if output_shift is None:
            decoder_network.get_layer(layer_name).set_weights([reduced_basis.T])
        else:
            decoder_network.get_layer(layer_name).set_weights([reduced_basis.T, output_shift])
        decoder_network.get_layer(layer_name).trainable = False

    return decoder_network


def create_reduced_network(rM, rZ, rU, pre_layers_m, pre_layers_z, post_layers, m_normalization, z_normalization, u_normalization):

    # Instance reduced coefficients of m as input layer
    m_red_input = tf.keras.layers.Input(shape = (rM, ), name = 'm_reduced')
    z_red_input = tf.keras.layers.Input(shape = (rZ, ), name = 'z_reduced')

    # Apply normalization operation
    m_red = m_red_input
    z_red = z_red_input

    if m_normalization is not None:
        m_mean, m_sd = m_normalization
        m_red = (m_red - m_mean)/m_sd

    if z_normalization is not None:
        z_mean, z_sd = z_normalization
        z_red = (z_red- z_mean)/z_sd

    # Pre concatenation layers
    for i, layer in enumerate(pre_layers_m):
        m_red = tf.keras.layers.Dense(layer, activation='softplus', name='pre_layer_m%d' %(i))(m_red)

    for i, layer in enumerate(pre_layers_z):
        z_red = tf.keras.layers.Dense(layer, activation='softplus', name='pre_layer_z%d' %(i))(z_red)

    # Concatenate inputs
    x_nn = tf.keras.layers.concatenate([m_red, z_red],name = 'concat')
    # print(x_nn)

    # Post concatenation layers
    for i, layer in enumerate(post_layers):
        x_nn= tf.keras.layers.Dense(layer, activation='softplus', name='layer%d' %(i))(x_nn)


    # Define the reduced network (rM x rZ -> dU)

    reduced_output = tf.keras.layers.Dense(rU, use_bias=False, name='reduced_output')(x_nn)
    if u_normalization is not None:
        u_mean, u_sd = u_normalization
        reduced_output = u_sd * reduced_output + u_mean

    reduced_network = tf.keras.models.Model([m_red_input, z_red_input], reduced_output)
    return reduced_network


def create_multi_input_reduced_basis_network(dM, dZ, dU, rM, rU,
            m_projector=None, m_shift=None, u_basis=None, u_shift=None,
            pre_layers_m=[], pre_layers_z=[], post_layers=[200, 200],
            m_normalization=None, z_normalization=None, u_normalization=None):
    """
    Create all components of the multi input reduced basis network

    - :code: `dM` input dimension for uncertain parameter
    - :code: `dZ` input dimension for optimization parameter
    - :code: `dU` output dimension 
    - :code: `rM` reduced input dimension for uncertain parameter
    - :code: `rU` reduced output dimension
    - :code: `m_projector` reduced projector for input
    - :code: `m_shift` shift (centering) for input 
    - :code: `u_basis` reduced basis for centered output
    - :code: `u_shift` shift (centering) for output 
    - :code: `pre_layers_m` list of layer widths for the pre-concatenation layers in m
    - :code: `pre_layers_z` list of layer widths for the pre-concatenation layers in z
    - :code: `post_layers` list of layer widths for the post-concatenation layers
    - :code: `m_normalization` [mean, sd] for reduced m inputs
    - :code: `z_normalization` [mean, sd] for reduced z inputs
    - :code: `u_normalization` [mean, sd] for reduced u outputs
    """
    if m_projector is not None:
        assert m_projector.shape[0] == dM
        assert m_projector.shape[1] == rM

    if u_basis is not None:
        assert u_basis.shape[0] == dU
        assert u_basis.shape[1] == rU

    reduced_network = create_reduced_network(rM, dZ, rU, pre_layers_m, pre_layers_z, post_layers, m_normalization, z_normalization, u_normalization)
    encoder_network = create_encoder_network(dM, rM, m_projector, input_shift=m_shift)
    decoder_network = create_decoder_network(rU, dU, u_basis, output_shift=u_shift)

    m_input = tf.keras.layers.Input(shape=(dM, ), name='m_input')
    z_input = tf.keras.layers.Input(shape=(dZ, ), name='z_input')

    z_red = z_input
    m_red = encoder_network(m_input)
    u_red = reduced_network([m_red, z_red])
    u_output = decoder_network(u_red)

    full_network = tf.keras.models.Model([m_input, z_input], u_output)
    return encoder_network, reduced_network, decoder_network, full_network


class MultiInputReducedBasisNetwork:
    """
    Class for a multi input reduced basis network mapping from dM x dZ -> dU
    with an internal network in reduced space rM x dZ -> rU
    """
    def __init__(self, dM, dZ, dU, rM, rU,
                 m_projector=None, m_shift=None, u_basis=None, u_projector=None, u_shift=None,
                 pre_layers_m=[], pre_layers_z=[], post_layers=[200, 200],
                 m_normalization=None, z_normalization=None, u_normalization=None):
        """
        """
        self.dM = dM
        self.dZ = dZ
        self.dU = dU
        self.rM = rM
        self.rU = rU
        self.pre_layers_m = pre_layers_m
        self.pre_layers_z = pre_layers_z
        self.post_layers = post_layers
        self.m_normalization = m_normalization
        self.z_normalization = z_normalization
        self.u_normalization = u_normalization

        self._m_shift = m_shift
        self._u_shift = u_shift 

        assert (u_basis is not None) or (u_projector is not None)

        if u_projector is None:
            self._u_projector = u_basis
        else:
            self._u_projector = u_projector

        self.encoder_network, self.reduced_network, self.decoder_network, self.full_network = create_multi_input_reduced_basis_network(
            dM, dZ, dU, rM, rU, 
            m_projector=m_projector, m_shift=m_shift, u_basis=u_basis, u_shift=u_shift, 
            pre_layers_m=pre_layers_m, pre_layers_z=pre_layers_z, post_layers=post_layers,
            m_normalization=m_normalization, z_normalization=z_normalization, u_normalization=u_normalization
        )

    def m_projector(self):
        return self.encoder_network.get_weights()[0]

    def u_basis(self):
        return self.decoder_network.get_weights()[0].T

    def u_projector(self):
        return self._u_projector

    def u_shift(self):
        return self._u_shift

    def project_m(self, m_data):
        return self.encoder_network.predict(m_data)

    def project_u(self, u_data):
        return u_data @ self._u_projector

    def orthogonalize_u_basis(self):
        u_basis = self.decoder_network.weights[0].numpy().T
        orth_basis, _ = np.linalg.qr(u_basis)
        self.decoder_network.set_weights([orth_basis])


    def save_metadata(self, directory):
        metadata = dict()
        metadata['dM'] = self.dM
        metadata['dZ'] = self.dZ
        metadata['dU'] = self.dU
        metadata['rM'] = self.rM
        metadata['rU'] = self.rU
        metadata['pre_layers_m'] = self.pre_layers_m
        metadata['pre_layers_z'] = self.pre_layers_z
        metadata['post_layers'] = self.post_layers

        with open('%s/metadata.p' %(directory), 'wb') as f:
            pickle.dump(metadata, f)


    def save_weights(self, directory):
        weights = {'reduced_network' : {}, 'encoder_network' : {}, 'decoder_network' : {}}
        # Save weights for each component
        for layer in self.reduced_network.layers:
            weights['reduced_network'][layer.name] = layer.get_weights()

        weights['reduced_network']['m_normalization'] = self.m_normalization
        weights['reduced_network']['z_normalization'] = self.z_normalization
        weights['reduced_network']['u_normalization'] = self.u_normalization

        for layer in self.encoder_network.layers:
            weights['encoder_network'][layer.name] = layer.get_weights()

        for layer in self.decoder_network.layers:
            weights['decoder_network'][layer.name] = layer.get_weights()

        weights['u_projector'] = self._u_projector 
        weights['u_shift'] = self._u_shift
        weights['m_shift'] = self._m_shift

        with open('%s/weights.p' %(directory), 'wb') as f:
            pickle.dump(weights, f)


    def freeze_output_basis(self):
        output_layer = self.decoder_network.layers[1]
        self.decoder_network.get_layer(output_layer.name).trainable = False


    def unfreeze_output_basis(self):
        output_layer = self.decoder_network.layers[1]
        self.decoder_network.get_layer(output_layer.name).trainable = True



def load_multi_input_reduced_basis_network(directory):
    """
    Load a multi-input reduced basis network from saved directory
    with `metadata.p` and `weights.p`
    """

    with open('%s/metadata.p' %(directory), 'rb') as f:
        metadata = pickle.load(f)

    with open('%s/weights.p' %(directory), 'rb') as f:
        weights = pickle.load(f)

    # Backward compatibility 
    if 'u_shift' in weights.keys():
        u_shift = weights['u_shift'] 
    else:
        u_shift = None

    if 'm_shift' in weights.keys():
        m_shift = weights['m_shift'] 
    else:
        m_shift = None

    rbnet = MultiInputReducedBasisNetwork(metadata['dM'], metadata['dZ'], metadata['dU'],
        metadata['rM'], metadata['rU'], 
        m_projector=None, m_shift=m_shift, 
        u_basis=None, u_shift = u_shift,
        u_projector = weights['u_projector'], 
        pre_layers_m=metadata['pre_layers_m'],
        pre_layers_z=metadata['pre_layers_z'],
        post_layers=metadata['post_layers'],
        m_normalization=weights['reduced_network']['m_normalization'],
        z_normalization=weights['reduced_network']['z_normalization'],
        u_normalization=weights['reduced_network']['u_normalization'])

    for layer in rbnet.reduced_network.layers:
        if layer.name.startswith('layer'):
            layer.set_weights(weights['reduced_network'][layer.name])
        if layer.name.startswith('pre_layer'):
            layer.set_weights(weights['reduced_network'][layer.name])
        if layer.name.startswith('reduced_output'):
            layer.set_weights(weights['reduced_network'][layer.name])

    for layer in rbnet.decoder_network.layers:
        layer.set_weights(weights['decoder_network'][layer.name])

    for layer in rbnet.encoder_network.layers:
        layer.set_weights(weights['encoder_network'][layer.name])

    return rbnet



def load_multi_input_reduced_basis_network_data(directory):
    with open('%s/metadata.p' %(directory), 'rb') as f:
        metadata = pickle.load(f)

    with open('%s/weights.p' %(directory), 'rb') as f:
        weights = pickle.load(f)

    return metadata, weights


def build_multi_input_reduced_basis_network_from_data(metadata, weights):
    """
    Build a multi-input reduced basis network from dictionaries
    metadata and weights 
    """

    # Backward compatibility 
    if 'u_shift' in weights.keys():
        u_shift = weights['u_shift'] 
    else:
        u_shift = None

    if 'm_shift' in weights.keys():
        m_shift = weights['m_shift'] 
    else:
        m_shift = None

    rbnet = MultiInputReducedBasisNetwork(metadata['dM'], metadata['dZ'], metadata['dU'],
        metadata['rM'], metadata['rU'], 
        m_projector=None, m_shift=m_shift, 
        u_basis=None, u_shift = u_shift,
        u_projector = weights['u_projector'], 
        pre_layers_m=metadata['pre_layers_m'],
        pre_layers_z=metadata['pre_layers_z'],
        post_layers=metadata['post_layers'],
        m_normalization=weights['reduced_network']['m_normalization'],
        z_normalization=weights['reduced_network']['z_normalization'],
        u_normalization=weights['reduced_network']['u_normalization'])

    for layer in rbnet.reduced_network.layers:
        if layer.name.startswith('layer'):
            layer.set_weights(weights['reduced_network'][layer.name])
        if layer.name.startswith('pre_layer'):
            layer.set_weights(weights['reduced_network'][layer.name])
        if layer.name.startswith('reduced_output'):
            layer.set_weights(weights['reduced_network'][layer.name])

    for layer in rbnet.decoder_network.layers:
        layer.set_weights(weights['decoder_network'][layer.name])

    for layer in rbnet.encoder_network.layers:
        layer.set_weights(weights['encoder_network'][layer.name])

    return rbnet

