"""
This version aligns with Deep Neural Decision Forests in that the splits are always binary
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import tensorflow as tf

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from keras import backend as K
from keras import initializers, layers

from keras.utils import to_categorical
from keras.constraints import non_neg, max_norm

from keras.initializers import Zeros
from keras.constraints import Constraint

import os
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers

import tensorflow as tf

from keras.layers import Dense, Flatten, Input, Lambda
from keras.models import Model

from keras import backend as K
from keras import initializers, layers
import keras

from keras.utils import to_categorical
from keras.constraints import min_max_norm, non_neg, max_norm
from keras.callbacks import Callback
import time
from datetime import datetime

class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, batch, logs={}):
        # write stuff to disc here...
        self.times.append(time.time() - self.epoch_time_start)


def split_pred(df, label):
    return df[[x for x in df.columns if x != label]], df[label]


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 18:
        lr *= 0.5e-3
    elif epoch > 16:
        lr *= 1e-3
    elif epoch > 12:
        lr *= 1e-2
    elif epoch > 8:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

import tensorflow as tf

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

from keras import backend as K
from keras import initializers, layers

from keras.utils import to_categorical
from keras.constraints import non_neg, max_norm

from keras.initializers import Zeros
from keras.constraints import Constraint
from keras.regularizers import l2


class ShrinkageConstraint(Constraint):

    def __init__(self, axis=0):
        self.axis = axis    
    
    def __call__(self, w):
        # apply non negative
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        # apply max value to be 1
        w *= K.cast(K.less_equal(w, 1.), K.floatx())
        return w

class ShrinkageFactor(layers.Layer):
    """
    This is the sigma object in the algorithm 1 by Beygelzimer (Online Gradient Boosting)
    """
    def __init__(self, step_size, is_trainable=True, **kwargs):
        self.step_size = step_size
        self.is_trainable = is_trainable
        super(ShrinkageFactor, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='highway', 
                                 shape=(1, 1),
                                 initializer=initializers.Zeros(),
                                 constraint=ShrinkageConstraint(),
                                 regularizer=l2(0.01),
                                 trainable=self.is_trainable)
        self.count = K.variable(0, name="epoch")
        super(ShrinkageFactor, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, x):
        return (1-self.step_size*self.W)*x
    
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape


class HighwayWeights(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(HighwayWeights, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='highway', 
                                 shape=(1, self.output_dim),
                                 initializer='uniform',
                                 constraint=non_neg(),
                                 trainable=True)
        super(HighwayWeights, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # if isinstance(x, list):
        #     if len(x) != K.int_shape(x[0][1]):
        #         raise Exception("The number of weights is a bit off...")
        W = self.W[0][:]
        W = W + K.epsilon()
        W = W/K.sum(W)
        W = K.abs(W)
        self.W = W
        
        output = W[0] * 0.0
        for idx in range(self.output_dim):
            output = output + W[idx]*x[idx]
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape


class BaseTree(object):
    def build_tree(self, depth=2):
        """
        builds the adjancey list up to depth of 2
        """
        total_nodes = np.sum([2**x for x in range(depth)])
        nodes = list(range(total_nodes))
        nodes_per_level = np.cumsum([2**x for x in range(depth-1)])
        nodes_level = [x.tolist() for x in np.array_split(nodes, nodes_per_level)]
        # path_list = list(itertools.product(*nodes_level))
        # return path_list
        adj_list = dict((idx, []) for idx in nodes)
        for fr in nodes_level[:-1]:
            for i in fr:
                i_list = adj_list.get(i, [])
                # the connected nodes always follows this pattern
                i_list.append(i*2 + 1)
                i_list.append(i*2 + 2)
                adj_list[i] = i_list
        return adj_list
    
    def calculate_routes(self, adj_list=None):
        """
        Calculates routes given a provided adjancency list,
        assume that root node is always 0.
        Assume this is a binary tree as well...
        Test cases:
        {0:[1, 2], 1:[], 2:[]} --> [(0, 0), (1, 0),
                                    (0, 0), (1, 1),
                                    (0, 1), (2, 0),
                                    (0, 1), (2, 1)]
        {0:[1], 1:[2], 2:[]}   --> [(0, 0), (1, 0), (2, 0),
                                    (0, 0), (1, 0), (2, 1),
                                    (0, 0), (1, 1),
                                    (0, 1)]
        calculate_routes({0:[1,2], 1:[], 2:[]})
        calculate_routes({0:[1], 1:[2], 2:[]})
        """
        if adj_list is None:
            adj_list = self.build_tree(3)
        
        def get_next(path):
            next_paths = adj_list[path[-1]]
            if len(next_paths) > 0:
                for p in next_paths:
                    get_next(path + [p])
            else:
                all_paths.append(path)

        all_paths = []
        get_next([0])

        # convert paths to indices...
        path_indx = []
        for path in all_paths:
            cur_path = []
            for cur_node, nxt_node in zip(path, path[1:]+[None]):
                # print(cur_node, nxt_node)
                pos_dir = np.array(sorted(adj_list[cur_node]))
                pos_idx = np.argwhere(pos_dir==nxt_node).flatten().tolist()
                if len(pos_idx) > 0 and len(pos_dir) == 2:  # i.e. has 2 children
                    cur_path.append((cur_node, pos_idx[0]))
                elif len(pos_idx) > 0 and len(pos_dir) == 1:  # i.e. has 1 child
                    path_indx.append(cur_path + [(cur_node, 1)])  # then it will have a leaf!
                    cur_path.append((cur_node, pos_idx[0]))
                elif nxt_node is not None:
                    cur_path.append((cur_node, pos_dir.shape[0]))
                else:
                    path_indx.append(cur_path + [(cur_node, 0)])
                    path_indx.append(cur_path + [(cur_node, 1)])
        return path_indx


class Tree(BaseTree):
    """
    Tree object to help abstract out some of the methods that are commonly used.
    Also used to help figure out how to maintain state around pruning and grafting nodes
    
    Usage:
    tt = Tree().graft()
    tt.plot()
    """
    def __init__(self, depth=3, nodes=None, tree=None, previous_state={}):
        self.depth = depth
        self.nodes = nodes if nodes is not None else np.sum([2**x for x in range(self.depth)])
        self.tree = tree if tree is not None else self.build_tree(self.depth)
        self.previous_state = previous_state
        self.update()
        self.min_nodes = len(list(self.tree.keys()))
    
    def plot(self):
        # plots using networkx
        # %matplotlib inline
        import networkx as nx
        ndim = np.max(list(self.tree.keys()))+1
        adj_matrix = np.zeros((ndim, ndim))
        
        for key, val in self.tree.items():
            for v in val:
                adj_matrix[key, v] = 1
        
        G = nx.from_numpy_matrix(adj_matrix, create_using=nx.MultiDiGraph())
        nx.draw(G, with_labels=True)
        
    def update_route(self):
        self.route = self.calculate_routes(self.tree)
    
    def update_tree(self):
        # updates the tree so that there are no "missing" nodes,
        # i.e. if it is:
        # {0:[1, 3], 1:[], 3:[]} --> {0:[1, 2], 1:[], 2:[]}
        curr_keys = sorted(list(self.tree.keys()))
        # update tree and fine what is missing...
        old_new_mapping = dict(zip(curr_keys, range(len(curr_keys))))
        self.tree = dict([(old_new_mapping[key], [old_new_mapping[x] for x in val]) for key, val in self.tree.items()])
    
    def update_nodes(self):
        # update the nodes to the max key
        self.nodes = np.max(list(self.tree.keys()))
    
    def get_leaves(self):
        return [key for key, val in self.tree.items() if len(val) == 0]
    
    def get_parent(self, child):
        parent_list = [key for key, val in self.tree.items() if child in val]
        #assert len(parent_list) == 1
        return parent_list[0]
    
    def get_prune_exceptions(self):
        # returns nodes + children if it just got grafted
        if self.previous_state.get('state', '') == 'graft':
            return self.previous_state.get('node', [])
        return []
    
    def get_graft_exceptions(self):
        # returns node that just got pruned if applicable
        if self.previous_state.get('state', '') == 'prune':
            return self.previous_state.get('node', [])
        return []
    
    def update(self):
        self.update_tree()
        self.update_route()
        self.update_nodes()
    
    def prune(self, update=False):
        """
        This updates the number of nodes, routes etc and randomly prunes a node
        """
        if len(self.tree.keys()) <= self.min_nodes:
            print("Tree is already of depth ~2, will not prune")
            return self.graft(update)
        import random
        black_list = self.get_prune_exceptions()
        prune_leaf = random.choice([x for x in self.get_leaves() if x not in black_list])
        # to prune this leaf remove the key from dictionary and remove it from any val which has it...
        parent_leaf = self.get_parent(prune_leaf)
        
        # update tree...
        new_tree = self.tree.copy()
        prune_leaves = new_tree[parent_leaf]
        for l in prune_leaves:
            new_tree.pop(l, None)
        
        # parent is now child
        new_tree[parent_leaf] = []
        new_state = {'state': 'prune', 'node': [parent_leaf]}
        if update:
            self.tree = new_tree
            self.update()
            self.previous_state = new_state.copy()
            return self
        else:
            return new_tree, new_state
        
    def graft(self, update=False):
        """
        This updates the number of nodes, routes etc and randomly graft a node which has 0 or 1 children
        """
        import random
        black_list = self.get_graft_exceptions()
        grow_leaf = random.choice([k for k, v in self.tree.items() if k not in black_list and len(v) <= 1])
        
        # update tree...get the max node...
        max_node = np.max(list(self.tree.keys()))
        
        new_tree = self.tree.copy()
        child_list = new_tree[grow_leaf]
        
        if len(child_list) == 0:
            add_nodes = [max_node+1, max_node+2]
            new_tree[grow_leaf] = add_nodes
            new_tree[max_node+1] = []
            new_tree[max_node+2] = []
        elif len(child_list) == 1:
            add_nodes = child_list + [max_node+1]
            new_tree[grow_leaf] = add_nodes
            new_tree[max_node+1] = []
        else:
            raise Exception("Child list chosen for should be of length 1 or 2, got child list: {}".format(child_list))
        
        new_state = {'state': 'graft', 'node': add_nodes + [grow_leaf]}
        if update:
            self.tree = new_tree
            self.update()
            self.previous_state = new_state.copy()
            return self
        else:
            return new_tree, new_state


class DecisionTreeNode(layers.Layer):
    """
    This constructs a decision tree, that is capable of searching deep through usage of
    stacking/bayesian model averaging for model selection.
    Parameters
    ----------
    depth: the depth that the decision tree will allow you to route to
    nodes: number of nodes to initalise, if not given, it is derived from depth
           parameter (full binary tree)
    kernel_initalizer: see Keras documentation
    
    Usage
    -----
    
    ```
    X, y_ = make_classification(n_classes=2, n_informative=5)
    y = to_categorical(y_)
    
    # try with our custom layer...
    main_input = Input(shape=(20,), name='main_input')
    
    # this is where the decision trees are learned
    tree_nodes = DecisionTreeNode(depth=4, name='decision_tree')(main_input)
    flatten = Flatten()(tree_nodes)
    pred_out = Dense(2, activation='softmax')(flatten)
    
    model = Model(inputs=[main_input], outputs=[pred_out])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X, y, epochs=100, verbose=0)
    print(pd.DataFrame(hist.history).iloc[-1])
    ```
    """
    def __init__(self, depth=3, nodes=None, 
                 initializer='glorot_uniform', 
                 regularizer=None,
                 constraint=max_norm(),
                 **kwargs):
        self.depth = depth
        self.nodes = nodes if nodes is not None else np.sum([2**x for x in range(self.depth)])        
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraint
        super(DecisionTreeNode, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input should be shape (2, feats, nodes)
        self.W = self.add_weight(shape=[2, input_shape[1], self.nodes],
                                 initializer=self.initializer,
                                 constraint=self.constraint,
                                 regularizer=self.regularizer,
                                 name='tree_nodes')
        self.built = True
    
    def call(self, inputs, training=None):
        """
        #inputs_expand = K.expand_dims(inputs, 2)
        #inputs_tiled = K.tile(inputs_expand, [1, 1, self.trees])
        #inputs_hat = K.dot(inputs_tiled, self.W)
        """        
        # perform multiplication.
        # inputs should be:(instances, 1, feats)
        inputs_expand = K.expand_dims(inputs, 1)
        
        x_out_k = K.dot(inputs_expand, self.W)
        x_out_kt = K.permute_dimensions(x_out_k, [0, 1, 3, 2])
        # ignore for now...
        # return a slice, as the 2nd index is single
        assert K.int_shape(x_out_kt)[1] == 1
        # output is: (None, trees, 2)
        x_out_kt = x_out_kt[:, 0, :, :]
        
        try:
            c = tf.nn.softmax(x_out_kt, axis=-1)
        except Exception as e:
            c = tf.nn.softmax(x_out_kt, dim=-1)
        return c
    
    def compute_output_shape(self, input_shape):
        # output is: (None, trees, 2)
        # return (input_shape[0], self.trees, self.n_class)
        return (input_shape[0], self.nodes, 2)

    def get_config(self):
        config = {
            'nodes': self.nodes,
            'depth': self.depth
        }
        base_config = super(DecisionTreeNode, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecisionTreeRouting(layers.Layer):
    """
    This constructs a decision tree, that is capable of searching deep through usage of 
    stacking/bayesian model averaging for model selection.
    Parameters
    ----------
    depth: the depth that the decision tree will allow you to route to
    route: all possible routes to the leaves that this particular tree can take,
           if not given then a full binary tree is assumed to be provided
    
    Usage
    -----
    
    ```
    X, y_ = make_classification(n_classes=2, n_informative=5)
    y = to_categorical(y_)
    
    # try with our custom layer...
    main_input = Input(shape=(20,), name='main_input')
    
    # this is where the decision trees are learned
    tree_nodes = DecisionTreeNode(depth=4, name='decision_tree')(main_input)
    tree_route = DecisionTreeRouting(depth=4)([main_input, tree_nodes])
    flatten = Flatten()(tree_route)
    pred_out = Dense(2, activation='softmax')(flatten)
    
    model = Model(inputs=[main_input], outputs=[pred_out])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X, y, epochs=100, verbose=0)
    print(pd.DataFrame(hist.history).iloc[-1])
    ```
    """
    def __init__(self, depth=3, route=None, **kwargs):
        self.depth = depth
        self.route = route if route is not None else Tree().calculate_routes(Tree().build_tree(self.depth))
        self.leaves = self.get_leaves(self.route)
        super(DecisionTreeRouting, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.built = True
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'route': self.route,
            'leaves': self.leaves
        }
        base_config = super(DecisionTreeRouting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def get_leaves(self, routes):
        return len(routes)

    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            inputs, c = inputs
        else:
            raise Exception("Inputs are expected to be a list being: <Input, Node Probability>, got {}".format(inputs))
        
        def get_route_proba(path):
            from functools import reduce
            # m_final = functools.reduce(lambda x, y: tf.multiply(x, y), your_matrix_list)
            path_slice = [c[:, node, decision] for node, decision in path]
            route_proba = reduce(lambda x, y: tf.multiply(x, y), path_slice)
            return route_proba
        
        def get_proba(route_paths):
            """
            Simulate an input data frame into the routing...
            """
            input_proba = []
            for path in route_paths:
                route_proba = get_route_proba(path)
                route_tiled = K.tile(K.expand_dims(route_proba, 1), [1, K.int_shape(inputs)[1]])
                assert K.int_shape(route_tiled) == K.int_shape(inputs)
                input_proba.append(route_tiled)
            return input_proba
        
        input_proba = get_proba(self.route)
        
        input_out = []
        for proba in input_proba:
            input_out_ = inputs * proba
            input_out.append(input_out_)
        
        proba_output = tf.stack(input_out, axis=1)
        proba_output = tf.transpose(proba_output, [0, 2, 1])
        return proba_output

    def compute_output_shape(self, input_shape):
        # assume it is a list
        input_shape_ = list(input_shape[0])  # assume input is a list
        return (input_shape_[0], input_shape_[1], self.leaves)



class DecisionPredRouting(layers.Layer):
    """
    This constructs a decision tree, that is capable of searching deep through usage of 
    stacking/bayesian model averaging for model selection.
    Parameters
    ----------
    depth: the depth that the decision tree will allow you to route to
    route: all possible routes to the leaves that this particular tree can take,
           if not given then a full binary tree is assumed to be provided
    
    Usage
    -----
    
    ```
    X, y_ = make_classification(n_classes=2, n_informative=5)
    y = to_categorical(y_)
    
    # try with our custom layer...
    main_input = Input(shape=(20,), name='main_input')
    
    # this is where the decision trees are learned
    tree_nodes = DecisionTreeNode(depth=4, name='decision_tree')(main_input)
    tree_route = DecisionTreeRouting(depth=4)([main_input, tree_nodes])
    flatten = Flatten()(tree_route)
    pred_out = Dense(2, activation='softmax')(flatten)
    
    model = Model(inputs=[main_input], outputs=[pred_out])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X, y, epochs=100, verbose=0)
    print(pd.DataFrame(hist.history).iloc[-1])
    ```
    """
    def __init__(self, depth=3, route=None, **kwargs):
        self.depth = depth
        self.route = route if route is not None else Tree().calculate_routes(Tree().build_tree(self.depth))
        self.leaves = self.get_leaves(self.route)
        super(DecisionPredRouting, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.built = True
    
    def get_config(self):
        config = {
            'depth': self.depth,
            'route': self.route,
            'leaves': self.leaves
        }
        base_config = super(DecisionPredRouting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def get_leaves(self, routes):
        return len(routes)

    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            inputs, c = inputs
        else:
            raise Exception("Inputs are expected to be a list being: <Input, Node Probability>, got {}".format(inputs))
        
        def get_route_proba(path):
            from functools import reduce
            # m_final = functools.reduce(lambda x, y: tf.multiply(x, y), your_matrix_list)
            path_slice = [c[:, node, decision] for node, decision in path]
            route_proba = reduce(lambda x, y: tf.multiply(x, y), path_slice)
            return route_proba
        
        def get_proba(route_paths):
            """
            Simulate an input data frame into the routing...
            """
            input_proba = []
            for path in route_paths:
                route_proba = get_route_proba(path)
                route_tiled = K.tile(K.expand_dims(K.expand_dims(route_proba, 1), 1), [1, K.int_shape(inputs)[1], K.int_shape(inputs)[2]])
                assert K.int_shape(route_tiled) == K.int_shape(inputs)
                input_proba.append(route_tiled)
            return input_proba
        
        input_proba = get_proba(self.route)
        
        input_out = []
        for proba in input_proba:
            input_out_ = inputs * proba
            input_out.append(input_out_)
        
        proba_output = tf.stack(input_out, axis=1)
        #print("this", K.int_shape(proba_output))
        proba_mean = K.mean(proba_output, 1)
        proba_mean = K.mean(proba_mean, 1)
        #print("that", K.int_shape(proba_mean))
        # normalise...
        proba_mean = proba_mean/tf.norm(proba_mean, ord=1, axis=1, keepdims=True)
        #print("that2", K.int_shape(proba_mean))
        return proba_mean

    def compute_output_shape(self, input_shape):
        # assume it is a list
        input_shape_ = list(input_shape[0])  # assume input is a list
        return (input_shape_[0], input_shape_[2])


def dnrf2(x, tree_num=1, depth=4, num_class=2, dim_size=None, sample=0.7):
    # to do, implement random column sampling/filtering
    # via lambda layer or otherwise
    if dim_size is not None:
        sample_size = min(max(int(dim_size * sample), 1), dim_size)
        if sample_size != dim_size:
            def pred_shape(input_shape):
                shape = list(input_shape)
                return tuple([shape[0], sample_size])
            import random
            sel_cols = random.sample(range(dim_size), sample_size)
            mask = np.zeros(dim_size)
            for sel in sel_cols:
                mask[sel] = 1
            mask = mask.astype(bool)
            x = layers.Lambda(lambda x: tf.boolean_mask(x, mask, axis=1), output_shape=pred_shape)(x)
            #print(x)

    tree_nodes = DecisionTreeNode(depth=depth, name=f'decision_tree{tree_num}')(x)
    tree_route = DecisionTreeRouting(depth=depth, name=f'decision_route{tree_num}')([x, tree_nodes])
    #tree_flatten = Flatten()(tree_route)
    # leaf_layers = Lambda(lambda x: [K.slice(xx[:, :, idx] for idx in range(K.int_shape(tree_route)[-1])])(tree_route)
    def outputshape(input_shape):
        return [(input_shape[0], input_shape[1]) for _ in range(input_shape[2])]

    leaf_layers = layers.Lambda(lambda x: [tf.squeeze(y) for y in tf.split(x, [1 for _ in range(K.int_shape(x)[2])], axis=2)], output_shape=outputshape)(tree_route)
    pred_layer_tree = [Dense(num_class, activation='softmax', name=f'decision_leaf_{tree_num}_{idx}')(x) for idx, x in enumerate(leaf_layers)]

    #model = Model(inputs=[main_input], outputs=pred_layer_tree)
    #model.predict(X)

    def normalise_pred(x):
        x = tf.stack(x)
        x = tf.transpose(x, [1, 0, 2])
        #print(x)
        return x
        print(x)
        cl = K.sum(x, axis=0)
        cl = cl/tf.norm(cl, ord=1, axis=1, keepdims=True)
        return cl

    def normalise_pred_shape(input_shape):
        shape = list(input_shape[0])
        num_trees = len(input_shape)
        return tuple([shape[0], num_trees, shape[1]])

        shape = list(input_shape[0])
        return tuple([shape[0], 2])

    stack_pred = layers.Lambda(normalise_pred, output_shape=normalise_pred_shape, name=f'pred_layer_{tree_num}')(pred_layer_tree)
    test_out = DecisionPredRouting(depth=depth)([stack_pred, tree_nodes])
    #return [test_out]+pred_layer_tree
    return test_out

if __name__ == '__main__':
    X, y_ = make_classification(n_classes=2, n_informative=5)
    y = to_categorical(y_)
    # try with our custom layer...
    main_input = Input(shape=(20,), name='main_input')
    
    # this is where the decision trees are learned
    main_input = Input(shape=(20,), name='main_input')
    pred_out = dnrf2(main_input)
    model = Model(inputs=[main_input], outputs=[pred_out])
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    hist = model.fit(X, y, epochs=200, verbose=0)
    print(pd.DataFrame(hist.history).iloc[-1])



