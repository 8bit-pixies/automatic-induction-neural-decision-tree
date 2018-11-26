import os
import sys
import math
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

from keras import backend as K
from keras import initializers, layers
from keras.utils import to_categorical
from keras.constraints import non_neg, max_norm
from keras.initializers import Zeros
from keras.constraints import Constraint
from keras import regularizers

import tensorflow as tf

from decision_tree import *
from datetime import datetime

time_cb = TimingCallback()


X, y_ = make_classification() # may want to increase complexity here
y = to_categorical(y_)
# make sure you do a test validation split!

tree = Tree() # this keeps the state of the current decision tree...
input_dim = 20
dim_size = 20
nepochs = 5 # we use nepochs=20 in paper

num_class = 2
num_trees = 5
num_rounds = 3

save_dir = "temp"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


def gen_states(tree, tree_list=[0], target_idx=None, return_tree_list=False):
    def size_0(dct):
        for key, val in dct.items():
            if len(val) > 0:
                return False
        return True
    
    tree_index = max(tree_list)
    if target_idx is None:
        curr_list = [tree_index+1, tree_index+2, tree_index+3]
    else:
        curr_list = [tree_index+1, target_idx, tree_index+2]
    
    tree_list.extend(curr_list)
    d0, s0 = tree.prune()
    d1 = tree.tree.copy()
    d2, s2 = tree.graft()
    
    if size_0(d0):
        # reset
        d0 = Tree().tree.copy()    
    
    state_info = {'prune': (d0, curr_list[0]),
            'base': (d1, curr_list[1]),
            'graft': (d2, curr_list[2]),
            'state': {
                'prune': s0, 'graft': s2
            }}
    if return_tree_list:
        return state_info, tree_list, curr_list
    else:
        return state_info


def outputshape(input_shape):
    return [(input_shape[0], input_shape[1]) for _ in range(input_shape[2])]


def normalise_pred(x):
    x = tf.stack(x)
    x = tf.transpose(x, [1, 0, 2])
    return x


def normalise_pred_shape(input_shape):
    shape = list(input_shape[0])
    num_trees = len(input_shape)
    return tuple([shape[0], num_trees, shape[1]])


def softmax_tau(proba, tau=0.1):
    """
    This is a softmax which goes towards one-hot encoding overtime. 
    We want to decay tau from 1.0 to 0.1 roughly
    """
    from scipy.special import logit, expit
    out = expit(logit(proba)/tau)
    return out/np.sum(out)


def get_layer_weights(model, name='hwy', sample=False, tau=1.0):
    out = K.eval([x for x in model.layers if x.name == name][0].weights[0]).flatten()
    return normalise_weights(out, sample, tau)


def normalise_weights(out, sample=False, tau=1.0):
    out = np.abs(out)
    out = out/np.sum(out)
    if sample and tau >= 1.0:
        draw = np.random.choice(range(out.shape[0]), 1, p=out)
        return draw[0]
    elif sample:
        draw = np.random.choice(range(out.shape[0]), 1, p=softmax_tau(out, tau))
        return draw[0]
    elif tau >= 1.0:
        return out
    else:
        return softmax_tau(out, tau)


def calculate_routes(adj_list=None):
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
        raise Exception("Adj_list cannot be none")

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


def build_tree(main_input, tree, tree_list, indx, tree_number=0):
    """
    Builds a single decision tree, returns all the specs needed to preserve tree state...
    """
    tree_state, tree_list, curr_list = gen_states(tree, tree_list[:], indx, True)
    
    route0 = calculate_routes(tree_state['prune'][0])
    route1 = calculate_routes(tree_state['base'][0])
    route2 = calculate_routes(tree_state['graft'][0])
    
    nodes0 = list(tree_state['prune'][0].keys())
    nodes1 = list(tree_state['base'][0].keys())
    nodes2 = list(tree_state['graft'][0].keys())

    all_nodes = list(set(nodes0 + nodes1 + nodes2))
    tree_nodes_list = len(all_nodes)
    
    route_name0 = "t{}_tree_route{}".format(tree_number, tree_state['prune'][1])
    route_name1 = "t{}_tree_route{}".format(tree_number, tree_state['base'][1])
    route_name2 = "t{}_tree_route{}".format(tree_number, tree_state['graft'][1])

    pred_name0 = "t{}_pred_route{}".format(tree_number, tree_state['prune'][1])
    pred_name1 = "t{}_pred_route{}".format(tree_number, tree_state['base'][1])
    pred_name2 = "t{}_pred_route{}".format(tree_number, tree_state['graft'][1])

    # create custom regularization weights based on the routes that it will be taking...
    def l1_reg(weight_matrix, nodes=[nodes0, nodes1, nodes2]):
        # weight matrix is shape (2, feats, nodes)
        unweighted_reg = 0.02 * K.sum(K.abs(weight_matrix))
        if len(nodes) == 0:
            return unweighted_reg
        else:
            # determine weights by the routing logic...
            base_weight = 0.01/len(nodes)
            running_weight = 0.0
            for nds in nodes:
                normalizer = base_weight * (1.0/len(nds)) * (math.sqrt(len(nds))/math.sqrt(7))
                for nd in nds:
                    running_weight += normalizer * K.sum(K.abs(weight_matrix[:, :, nd]))
            return unweighted_reg-running_weight
    
    tree_nodes = DecisionTreeNode(nodes=tree_nodes_list, regularizer=l1_reg)(main_input)

    tree_r0 = DecisionTreeRouting(route=route0, name=route_name0)([main_input, tree_nodes])
    tree_r1 = DecisionTreeRouting(route=route1, name=route_name1)([main_input, tree_nodes])
    tree_r2 = DecisionTreeRouting(route=route2, name=route_name2)([main_input, tree_nodes])

    leaf_layers0 = layers.Lambda(lambda x: [tf.squeeze(y) for y in tf.split(x, [1 for _ in range(K.int_shape(x)[2])], axis=2)], output_shape=outputshape)(tree_r0)
    leaf_layers1 = layers.Lambda(lambda x: [tf.squeeze(y) for y in tf.split(x, [1 for _ in range(K.int_shape(x)[2])], axis=2)], output_shape=outputshape)(tree_r1)
    leaf_layers2 = layers.Lambda(lambda x: [tf.squeeze(y) for y in tf.split(x, [1 for _ in range(K.int_shape(x)[2])], axis=2)], output_shape=outputshape)(tree_r2)
    # As part of this step, we need to understand how we can general all leaf nodes; for all models, and identify leaf nodes which are shared and which ones are not.
    def get_leafs(tree_info):
        '''
        retrieves the name of the leafs, which are:
            name of the parent + name of the route (left or right)
        
        we add random datetime so that they are retrained everyrun - it doesn't make sense that it is updated in tandem
        '''
        tree = Tree(tree=tree_info)
        leaves = tree.get_leaves()
        parent_leaves = ["p{}_l{}_t{}_{}".format(tree.get_parent(idx), idx, tree_number, datetime.now().strftime("%H%M%S")) for idx in leaves]
        return parent_leaves

    leaf_names0 = get_leafs(tree_state['prune'][0])
    leaf_names1 = get_leafs(tree_state['base'][0])
    leaf_names2 = get_leafs(tree_state['graft'][0])

    leaf_names_all = set(leaf_names0 + leaf_names1 + leaf_names2)
    pred_layer = {nm:Dense(num_class, activation='softmax', name=nm) for nm in leaf_names_all}

    pred_layer_tree0 = [pred_layer[nm](x) for nm, x in zip(leaf_names0, leaf_layers0)]
    pred_layer_tree1 = [pred_layer[nm](x) for nm, x in zip(leaf_names1, leaf_layers1)]
    pred_layer_tree2 = [pred_layer[nm](x) for nm, x in zip(leaf_names2, leaf_layers2)]

    stack_pred0 = layers.Lambda(normalise_pred, output_shape=normalise_pred_shape)(pred_layer_tree0)
    stack_pred1 = layers.Lambda(normalise_pred, output_shape=normalise_pred_shape)(pred_layer_tree1)
    stack_pred2 = layers.Lambda(normalise_pred, output_shape=normalise_pred_shape)(pred_layer_tree2)
    tree_d0 = DecisionPredRouting(route=route0)([stack_pred0, tree_nodes])
    tree_d1 = DecisionPredRouting(route=route1)([stack_pred1, tree_nodes])
    tree_d2 = DecisionPredRouting(route=route2)([stack_pred2, tree_nodes])

    highway_layer = HighwayWeights(output_dim=3, name='hwy{}'.format(tree_number))([tree_d0, tree_d1, tree_d2])
    return highway_layer, tree_state, tree_list, curr_list

tree_index = 0
forest = [Tree() for idx in range(num_trees)]
main_input = Input(shape=(dim_size,), name='main_input')
tree_listing = [build_tree(main_input, forest[idx], [0], None, idx) for idx in range(num_trees)]

#t0, tree_state0, tree_list0, curr_list0 = build_tree(main_input, forest[0], [0], None, 0)
#t1, tree_state1, tree_list1, curr_list1 = build_tree(main_input, forest[1], [0], None, 1)
#t2, tree_state2, tree_list2, curr_list2 = build_tree(main_input, forest[2], [0], None, 2)
#t3, tree_state3, tree_list3, curr_list3 = build_tree(main_input, forest[3], [0], None, 3)
#t4, tree_state4, tree_list4, curr_list4 = build_tree(main_input, forest[4], [0], None, 4)

def normalise_pred2(x):
    x = tf.stack(x)
    x = tf.transpose(x, [1, 0, 2])
    cl = K.sum(x, axis=1)
    cl = cl/tf.norm(cl, ord=1, axis=1, keepdims=True)
    return cl

def normalise_pred_shape2(input_shape):
    shape = list(input_shape[0])
    return tuple([shape[0], num_class])

stack_pred = layers.Lambda(normalise_pred2, output_shape=normalise_pred_shape2)([tl[0] for tl in tree_listing])
model = Model(inputs=[main_input], outputs=[stack_pred])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X, y, epochs=nepochs, verbose=0)
hist_df = pd.DataFrame(hist.history)
print(pd.DataFrame(hist.history).iloc[-1])

model.save_weights(os.path.join(save_dir, 'temp_model_s.h5'))


tau = 1.0
discount = 0.99

for iters in range(num_rounds):
    print("\n\nCurrent Iter: {}".format(iters))
    try:
        tau = tau * discount
        next_idx = [get_layer_weights(model, name='hwy{}'.format(idx), tau=tau, sample=True) for idx in range(num_trees)]
        
        #next_idx0 = get_layer_weights(model, name='hwy0', tau=tau, sample=True)
        #next_idx1 = get_layer_weights(model, name='hwy1', tau=tau, sample=True)
        #next_idx2 = get_layer_weights(model, name='hwy2', tau=tau, sample=True)
        #next_idx3 = get_layer_weights(model, name='hwy3', tau=tau, sample=True)
        #next_idx4 = get_layer_weights(model, name='hwy4', tau=tau, sample=True)
        actions = ['prune', 'base', 'graft']
        #print("Next idx: {}, action: {}".format(curr_list[next_idx], actions[next_idx]))
        
        #tree0 = Tree(tree=tree_state0[actions[next_idx0]][0])
        #tree1 = Tree(tree=tree_state1[actions[next_idx1]][0])
        #tree2 = Tree(tree=tree_state2[actions[next_idx2]][0])
        #tree3 = Tree(tree=tree_state3[actions[next_idx3]][0])
        #tree4 = Tree(tree=tree_state4[actions[next_idx4]][0])
        
        forest = [Tree(tree=tree_listing[idx][1][actions[next_idx[idx]]][0]) for idx in range(num_trees)]

        main_input = Input(shape=(dim_size,), name='main_input')
        tree_listing = [build_tree(main_input, forest[idx], tree_listing[idx][2], tree_listing[idx][3][next_idx[idx]], idx) for idx in range(num_trees)]
        
        #t0, tree_state0, tree_list0, curr_list0 = build_tree(main_input, tree0, tree_list0, curr_list0[next_idx0], 0)
        #t1, tree_state1, tree_list1, curr_list1 = build_tree(main_input, tree1, tree_list1, curr_list1[next_idx1], 1)
        #t2, tree_state2, tree_list2, curr_list2 = build_tree(main_input, tree2, tree_list2, curr_list2[next_idx2], 2)
        #t3, tree_state3, tree_list3, curr_list3 = build_tree(main_input, tree3, tree_list3, curr_list3[next_idx3], 3)
        #t4, tree_state4, tree_list4, curr_list4 = build_tree(main_input, tree4, tree_list4, curr_list4[next_idx4], 4)
        
        stack_pred = layers.Lambda(normalise_pred2, output_shape=normalise_pred_shape2)([tl[0] for tl in tree_listing])
        model = Model(inputs=[main_input], outputs=[stack_pred])

        model.load_weights(os.path.join(save_dir, 'temp_model_s.h5'), by_name=True)
        for idx in range(num_trees):
            model.get_layer('hwy{}'.format(idx)).set_weights([np.array([[0.25, 0.5, 0.25]])])
        
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        hist = model.fit(X, y, epochs=nepochs, verbose=0)
        pd_temp = pd.DataFrame(hist.history)
        print(pd.DataFrame(hist.history).iloc[-1])
        hist_df = pd.concat([hist_df, pd_temp])
        model.save_weights(os.path.join(save_dir, 'temp_model_s.h5'))
        # print("Highway: {}".format(K.eval(model.get_layer('hwy').weights)))
    except Exception as e:
        print(e)

hist_df.to_csv(os.path.join(save_dir, "rf200.csv"))

tau = tau * discount
next_idx = [get_layer_weights(model, name='hwy{}'.format(idx), tau=tau, sample=True) for idx in range(num_trees)]

#next_idx0 = get_layer_weights(model, name='hwy0', tau=tau, sample=True)
#next_idx1 = get_layer_weights(model, name='hwy1', tau=tau, sample=True)
#next_idx2 = get_layer_weights(model, name='hwy2', tau=tau, sample=True)
#next_idx3 = get_layer_weights(model, name='hwy3', tau=tau, sample=True)
#next_idx4 = get_layer_weights(model, name='hwy4', tau=tau, sample=True)
actions = ['prune', 'base', 'graft']
#print("Next idx: {}, action: {}".format(curr_list[next_idx], actions[next_idx]))

#tree0 = Tree(tree=tree_state0[actions[next_idx0]][0])
#tree1 = Tree(tree=tree_state1[actions[next_idx1]][0])
#tree2 = Tree(tree=tree_state2[actions[next_idx2]][0])
#tree3 = Tree(tree=tree_state3[actions[next_idx3]][0])
#tree4 = Tree(tree=tree_state4[actions[next_idx4]][0])

forest = [Tree(tree=tree_listing[idx][1][actions[next_idx[idx]]][0]) for idx in range(num_trees)]

main_input = Input(shape=(dim_size,), name='main_input')
tree_listing = [build_tree(main_input, forest[idx], tree_listing[idx][2], tree_listing[idx][3][next_idx[idx]], idx) for idx in range(num_trees)]

#t0, tree_state0, tree_list0, curr_list0 = build_tree(main_input, tree0, tree_list0, curr_list0[next_idx0], 0)
#t1, tree_state1, tree_list1, curr_list1 = build_tree(main_input, tree1, tree_list1, curr_list1[next_idx1], 1)
#t2, tree_state2, tree_list2, curr_list2 = build_tree(main_input, tree2, tree_list2, curr_list2[next_idx2], 2)
#t3, tree_state3, tree_list3, curr_list3 = build_tree(main_input, tree3, tree_list3, curr_list3[next_idx3], 3)
#t4, tree_state4, tree_list4, curr_list4 = build_tree(main_input, tree4, tree_list4, curr_list4[next_idx4], 4)

stack_pred = layers.Lambda(normalise_pred2, output_shape=normalise_pred_shape2)([tl[0] for tl in tree_listing])
model = Model(inputs=[main_input], outputs=[stack_pred])

model.load_weights(os.path.join(save_dir, 'temp_model_s.h5'), by_name=True)
for idx in range(num_trees):
    model.get_layer('hwy{}'.format(idx)).set_weights([np.array([[0.25, 0.5, 0.25]])])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
hist = model.fit(X, y, epochs=nepochs*10, verbose=0)
pd_temp = pd.DataFrame(hist.history)
print(pd.DataFrame(hist.history).iloc[-1])
hist_df = pd.concat([hist_df, pd_temp])
model.save_weights(os.path.join(save_dir, 'temp_model_s.h5'))

