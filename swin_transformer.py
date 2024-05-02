import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.activations import gelu


CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}


class MLP(tf.keras.layers.Layer):
    def __init__(self, input_features, hidden_features=None, output_features=None, dropout=0.0, prefix=''):
        super().__init__()
        if output_features is not None:
            output_features = output_features
        else:
            output_features = input_features
        
        if hidden_features is not None:
            hidden_features = hidden_features
        else:
            hidden_features = input_features
        
        self.hidden_layer_1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.hidden_layer_2 = Dense(output_features, name=f'{prefix}/mlp/fc2')
        self.dropout = Dropout(dropout)

    def call(self, x):
        value = self.hidden_layer_1(x)
        value = gelu(value)
        value = self.dropout(value)
        value = self.hidden_layer_2(value)
        value = self.dropout(value)

        return value

def window_partition(x, win_size):

    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, (H//win_size), win_size, (W//win_size), win_size, C])
    x = tf.transpose(x, perm=[0,1,3,2,4,5])

    partitioned_windows = tf.reshape(x, shape=[-1, win_size, win_size, C])

    return partitioned_windows

def window_reverse(partitioned_windows, win_size, H, W, C):

    x = tf.reshape(partitioned_windows, shape=[-1, (H//win_size), (W//win_size), win_size, win_size, C])
    x = tf.transpose(x, perm=[0,1,3,2,4,5])
    reversed_windows = tf.reshape(x, shape=[-1, H, W, C])

    return reversed_windows

class WindowAttentionHead(tf.keras.layers.Layer):

    def __init__(self, dimension, win_size, n_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, prefix=''):

        super().__init__()

        self.dimension=dimension
        self.win_size=win_size
        self.n_heads = n_heads

        head_dimension = dimension//n_heads

        if qk_scale is not None:
            self.scaling_factor = qk_scale
        else:
            self.scaling_factor = head_dimension**(-0.5)
        
        self.prefix=prefix
        
        self.qkv = Dense(dimension*3, use_bias=qkv_bias, name=f'{self.prefix}/attn/qkv')
        self.attention_dropout = Dropout(attn_drop)
        self.proj = Dense(dimension, name=f'{self.prefix}/attn/proj')
        self.proj_dropout = Dropout(proj_drop)
    
    def build(self, input_shape):
        self.relative_pos_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',shape=((2 * self.win_size[0] - 1) * (2 * self.win_size[1] - 1), self.n_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)
        

        vertical_coords = np.arange(self.win_size[0])
        horizontal_coords = np.arange(self.win_size[1])

        cartesian_grid = np.stack(np.meshgrid(vertical_coords, horizontal_coords, indexing='ij'))

        cartesian_flattened = cartesian_grid.reshape(2,-1)

        
        relative_coords = cartesian_flattened[:,:,None] - cartesian_flattened[:,None,:]
        relative_coords = relative_coords.transpose([1,2,0])
        relative_coords[:,:,0] += self.win_size[0]-1
        relative_coords[:,:,1] += self.win_size[1]-1
        relative_coords[:,:,0] *= 2*self.win_size[1]-1

        relative_pos_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_pos_index = tf.Variable(initial_value=tf.convert_to_tensor(relative_pos_index), trainable=False, 
                                              name=f'{self.prefix}/attn/relative_position_index')
        
        self.built = True

    def call(self, x, mask=None):

        _B, N, C = x.get_shape().as_list()

        qkv = tf.transpose(tf.reshape(self.qkv(x), shape=[-1,N,3, self.n_heads, C//self.n_heads]), perm=[2,0,3,1,4])

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q*self.scaling_factor
        attention = (q @ tf.transpose(k, perm=[0,1,3,2]))
        relative_position_bias = tf.gather(self.relative_pos_bias_table, tf.reshape(self.relative_pos_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[self.win_size[0]*self.win_size[1], self.win_size[0]*self.win_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2,0,1])

        attention = attention + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:

            n_W = mask.get_shape()[0]

            attention = tf.reshape(attention, shape=[-1, n_W, self.n_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                attention.dtype
            )

            attention = tf.reshape(attention, shape=[-1, self.n_heads, N, N])

            attention = tf.nn.softmax(attention, axis=-1)
        else:
            attention = tf.nn.softmax(attention, axis=-1)
        
        attention = self.attention_dropout(attention)

        x = tf.transpose((attention@v), perm=[0,2,1,3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
    
def drop_path(inputs, dropout_prob, training):
    if ((not training) or (dropout_prob==0.0)):
         return inputs
    
    keep_prob = 1.0 - dropout_prob

    random_connect_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
    random_connect_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_connect_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor

    return output

class DropPath(tf.keras.layers.Layer):
    def __init__(self, dropout_prob=None):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def call(self, x, training=None):
        return drop_path(x, self.dropout_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dimension, input_resolution, n_heads, win_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, dropout=0.0, 
                 attention_dropout=0.0, drop_path_prob=0.0, norm_layer = LayerNormalization, prefix=''):
        
        super().__init__()
        self.dimension = dimension
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.n_heads = n_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.win_size:
            self.shift_size=0
            self.win_size = min(self.input_resolution)
        
        assert 0<= self.shift_size < self.win_size, "shift size must be <= window size"
        self.prefix=prefix

        self.normalization_layer_1 = norm_layer(epsilon=1e-5)

        self.attention = WindowAttentionHead(dimension=dimension, win_size=(self.win_size, self.win_size), n_heads=n_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attention_dropout, proj_drop=dropout, prefix=self.prefix)

        self.drop_path = DropPath(drop_path_prob if drop_path_prob>0.0 else 0.0)

        self.normalization_layer_2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')

        mlp_hidden_dimension = int(dimension*mlp_ratio)

        self.MLP = MLP(input_features=dimension, hidden_features=mlp_hidden_dimension, dropout=dropout, prefix=self.prefix)

    def build(self, input_shape):

        if self.shift_size>0:
            H, W = self.input_resolution

            img_mask = np.zeros([1,H,W,1])

            height_slices = (slice(0, -self.win_size),
                                 slice(-self.win_size, -self.shift_size),
                                 slice(-self.shift_size, None))
            
            width_slices = (slice(0, -self.win_size),
                            slice(-self.win_size, -self.shift_size),
                            slice(-self.shift_size, None))
            
            i = 0

            for h in height_slices:
                for w in width_slices:
                    img_mask[:,h,w,:] = i
                    i += 1
            
            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.win_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.win_size*self.win_size])
            attention_mask = tf.expand_dims(mask_windows, axis=1)-tf.expand_dims(mask_windows, axis=2)

            attention_mask = tf.where(attention_mask!=0, -100.0, attention_mask)
            self.attention_mask = tf.Variable(initial_value=attention_mask, trainable=False, name=f'{self.prefix}/attn_mask')

        else:
            self.attention_mask = None
        
        self.built = True
    
    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        assert L== H*W, "wrong size of the input"

        sc = x
        x = self.normalization_layer_1(x)
        x = tf.reshape(x, shape=[-1,H,W,C])

        if self.shift_size>0:
            x_shifted = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1,2])
        else:
            x_shifted = x
        
        # partition_windows
        windows_x = window_partition(x_shifted, self.win_size)
        windows_x = tf.reshape(windows_x, shape=[-1, self.win_size*self.win_size, C])

        # Window Attention Head/ Shifted Window Attention Head
        attention_windows = self.attention(windows_x, mask=self.attention_mask)
        
        # merge the windows
        attention_windows = tf.reshape(attention_windows, shape=[-1, self.win_size, self.win_size, C])
        x_shifted = window_reverse(attention_windows, self.win_size, H=H, W=W, C=C)

        # reverse cyclic shift

        if self.shift_size>0:
            x = tf.roll(x_shifted, shift=[self.shift_size, self.shift_size],
                        axis=[1,2])
        else:
            x = x_shifted
        
        x = tf.reshape(x, shape=[-1, H*W, C])

        # skip-connection
        x = sc + self.drop_path(x)

        x = x + self.drop_path(self.MLP(self.normalization_layer_2(x)))

        return x

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dimension, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dimension = dimension
        self.reduction_layer = Dense(2*dimension, use_bias=False, name=f'{prefix}/downsample/reduction')
        self.normalization_layer = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        assert L==H*W, "Wrong input shape!"
        assert (H%2==0) and (W%2==0), f"Either H={H} or W={W} are not even"

        x = tf.reshape(x, shape=[-1,H,W,C])

        x1 = x[:,0::2, 0::2, :]
        x2 = x[:,1::2, 0::2, :]
        x3 = x[:,0::2, 1::2, :]
        x4 = x[:,1::2, 1::2, :]
        x = tf.concat([x1, x2, x3, x4], axis=-1)
        x = tf.reshape(x, shape=[-1, (H//2)*(W//2), 4*C])

        x = self.normalization_layer(x)
        x = self.reduction_layer(x)

        return x

class BasicLayer(tf.keras.layers.Layer):
    
    def __init__(self, dimension, input_resolution, depth, n_heads, win_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, dropout=0.0, 
                 attention_dropout=0.0, drop_path_prob=0.0, norm_layer = LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dimension
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.block = tf.keras.Sequential([
            SwinTransformerBlock(dimension=dimension, input_resolution=input_resolution, n_heads=n_heads, win_size=win_size, shift_size=0 if(i%2==0) else win_size//2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 dropout=dropout, attention_dropout=attention_dropout,drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob, norm_layer=norm_layer, prefix=f'{prefix}/blocks{i}') for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(input_resolution, dimension=dimension, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample=None 
        
    def call(self, x):
        x = self.block(x)

        if self.downsample is not None:
            x = self.downsample(x)
        
        return x
    
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, image_size=(224,224), patch_size=(4,4), input_channels=3, embed_dimension=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patch_resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]
        self.n_patches = self.patch_resolution[0]*self.patch_resolution[1]

        self.input_channels = input_channels
        self.embed_dimension = embed_dimension

        self.proj = Conv2D(embed_dimension, kernel_size=patch_size, strides=patch_size, name='proj')

        if norm_layer is not None:
            self.normalization_layer = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.normalization_layer = None
    
    def call(self, x):
        B, H, W, C = x.get_shape().as_list()

        assert H==self.image_size[0] and W==self.image_size[1], f"Input image size {H}x{W} does not match model input shape: {self.image_size[0]}x{self.image_size[1]}"

        x = self.proj(x)
        x = tf.reshape(x, shape=[-1, (H//self.patch_size[0])*(W//self.patch_size[0]), self.embed_dimension])

        if self.normalization_layer is not None:
            x = self.normalization_layer(x)

        return x
    
class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='SwinTransformer', include_top=False, image_size=(224,224), patch_size=(4,4), input_channels=3, num_classes=1000, embed_dimension=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 win_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, dropout_rate=0.0, attention_dropout_rate=0.0, drop_path_rate=0.1, 
                 norm_layer = LayerNormalization, absolute_positional_embedding=False, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top
        self.n_classes = num_classes
        self.n_layers = len(depths)
        self.embed_dimension = embed_dimension
        self.absolute_pos_embedding = absolute_positional_embedding
        self.patch_normalization = patch_norm
        self.n_features = int(embed_dimension*2**(self.n_layers-1))
        self.mlp_ratio = mlp_ratio

        # splitting the image into non overlapping patches
        self.patch_embedding = PatchEmbed(image_size=image_size, patch_size=patch_size, input_channels=input_channels, embed_dimension=embed_dimension, norm_layer=norm_layer if self.patch_normalization else None)
        n_patches = self.patch_embedding.n_patches
        patch_resolution = self.patch_embedding.patch_resolution
        self.patch_resolution = patch_resolution

        if self.absolute_pos_embedding:

            self.absolute_positional_embedding = self.add_weight('absolute_pos_embed',shape=(1, n_patches, embed_dimension), initializer=tf.initializers.Zeros())
        
        self.positional_dropout = Dropout(dropout_rate)

        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        self.basic_layers = tf.keras.Sequential([
            BasicLayer(dimension=int(embed_dimension*2**i_layer),
                       input_resolution=(patch_resolution[0]//(2**i_layer), patch_resolution[1]//(2**i_layer)),
                       depth=depths[i_layer],
                       n_heads = num_heads[i_layer],
                       win_size=win_size, 
                       mlp_ratio=self.mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=dropout_rate, 
                       drop_path_prob=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                       norm_layer=norm_layer,
                       downsample=PatchMerging if (i_layer<self.n_layers-1) else None, 
                       use_checkpoint=use_checkpoint, 
                       prefix=f'layers{i_layer}') for i_layer in range(self.n_layers)
        ])
        self.normalization_layer = norm_layer(epsilon=1e-5, name='norm')
        self.global_average_pooling = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None
        
    def forward_features(self,x):
        x = self.patch_embedding(x)
        if self.absolute_pos_embedding:
            x = x + self.absolute_positional_embedding
        x = self.positional_dropout(x)

        x = self.basic_layers(x)
        x = self.normalization_layer(x)
        x = self.global_average_pooling(x)

        return x
    
    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.head(x)
        
        return x

def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
    configurations = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name, include_top=include_top, num_classes=num_classes, 
        image_size=configurations['input_size'], 
        win_size= configurations['window_size'],
        embed_dimension=configurations['embed_dim'],
        depths=configurations['depths'],
        num_heads=configurations['num_heads']
    )
    net(tf.keras.Input(shape=(configurations['input_size'][0], configurations['input_size'][1], 3)))
    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_weights = tf.keras.utils.get_file(model_name, url, untar=True)
    else:
        pretrained_weights = pretrained
    
    if pretrained_weights:
        if tf.io.gfile.isdir(pretrained_weights):
            pretrained_weights = f'{pretrained_weights}/{model_name}.ckpt'
        
        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost'
            )
            net.load_weights(pretrained_weights, options=load_locally)
        else:
            net.load_weights(pretrained_weights)
    
    return net


