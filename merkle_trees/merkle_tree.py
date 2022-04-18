import hashlib
import numpy as np

def create_merkle_tree(D, C):
    n_hashes = int(len(D)/C)
    if n_hashes*C < len(D):
        n_hashes += 1
    n_nodes = 2*n_hashes -1 
#    T = [hashlib.new('sha1')]*n_nodes
    T = ['']*n_nodes

    leaf_start = n_hashes - 1
    print("Num hashes: " + str(n_hashes))
    print("Num nodes: " + str(n_nodes))
    print("Leaf start: " + str(leaf_start))
    for node in range(n_nodes-1, -1, -1):
#        print(node)
        if node >= leaf_start:
#            T[node] = hashlib.sha1(D[node-leaf_start:node-leaf_start+C].encode('ascii'))
            T[node] = hashlib.sha1(D[node-leaf_start:node-leaf_start+C].encode('ascii')).digest()
        else:
            child_l = 2*node+1
            child_r = 2*node+2
#            print("\t" + str(child_l))
#            print("\t" + str(child_r))
#            T[node] = hashlib.sha1(T[child_l].digest() + T[child_r].digest())
            T[node] = hashlib.sha1((T[child_l] + T[child_r])).digest()
#            T[node] = hashlib.sha1((T[child_l:child_r+1])).digest()
#        print(T[node].hexdigest())
#    print(len(T))
    print(type(T[0]))
    return T

def print_tree(T):
    n_nodes = len(T)
    counter = 2
#    print("Node: " + str(0) + "    (" + T[0].hexdigest() + ")")
    print("Node: " + str(0) + "    (" + T[0].hex() + ")")
    print("\n")
    for node in range(1, n_nodes):
#        print("Node: " + str(node) + "    (" + T[node].hexdigest() + ")")
        print("Node: " + str(node) + "    (" + T[node].hex() + ")")
        if(node == counter):
            print("\n")
            counter += 2*counter

def find_distinct_subtrees(T, id):
    Md = {}
    Ms = {}
    for node in range(len(T)):
        val = (node,node,id)
#        value = Md.get(T[node].hexdigest())
        value = Md.get(T[node])
        if value == None:
#            Md[T[node].hexdigest()] = val;
            Md[T[node]] = val;
        else:
            Ms[node] = value
    print("Distinct("+str(len(Md))+"): " + str(Md) + str('\n'))
    print("Shared("+str(len(Ms))+"): " + str(Ms) + str('\n'))
    return (Md, Ms)

def remove_subtree(T, root, Map, MapS):
    queue = []
    l_child = 2*root+1
    r_child = 2*root+2
    if l_child < len(T):
        queue.append(l_child)
    if r_child < len(T):
        queue.append(r_child)
    while len(queue) > 0:
        node = queue[0]
        queue.remove(queue[0])
        if T[node] in Map and node not in MapS:
            del Map[T[node]]
        l_child = 2*node+1
        r_child = 2*node+2
        if l_child < len(T):
            queue.append(l_child)
        if r_child < len(T):
            queue.append(r_child)

def deduplication(T, U):
    (Md, Ms) = find_distinct_subtrees(T, len(U))
    queue = []
    queue.append(0)
    while len(queue) > 0:
        node = queue[0]
        queue.remove(queue[0])
        if T[node] in Md:
            found = False
            for Mi in U:
                res = Mi.get(T[node])
                if res != None:
                    found = True
                    (n, m, id) = Md[T[node]]
                    (n_old, m_old, id_old) = Mi[T[node]]
                    remove_subtree(T, node, Md, Ms)
                    Md[T[node]] = (n, m_old, id_old)
                    break
            if not found:
                l_child = 2*node+1
                r_child = 2*node+2
                if l_child < len(T):
                    queue.append(l_child)
                if r_child < len(T):
                    queue.append(r_child)
    return (Md,Ms)

def find_hash(Md, Ms, node):
    search_node = node
    if node in Ms:
        (u,v,id) = Ms[node]
        search_node = u
    for key,val in Md.items():
        (u,v,id) = val
        if u == search_node:
            return key

def copy_subtree(T,U,V,hash,node,src,id):
    print("Copying subtree from tree " + str(id) + ", node " + str(src) + " to node " + str(node))
    nhashes = (len(T) + 1) / 2
    tree_node = node
    Md = U[id]
    Ms = V[id]
    tree_queue = []
    tree_queue.append(tree_node)
    queue = []
    queue.append(src)
    while len(queue) > 0:
        print("Queue: " + str(queue))
        print("Tree Queue: " + str(tree_queue))
        u = queue[0]
        queue.remove(queue[0])
        v = tree_queue[0]
        tree_queue.remove(tree_queue[0])
        print("Tree node: " + str(v) + ", old tree node: " + str(u))
        T[v] = find_hash(Md, Ms, u)
        l = 2*u+1
        r = 2*u+2
        if l <= len(T)-nhashes:
            queue.append(l)
        if r <= len(T)-nhashes:
            queue.append(r)
        l = 2*v+1
        r = 2*v+2
        if l <= len(T)-nhashes:
            tree_queue.append(l)
        if r <= len(T)-nhashes:
            tree_queue.append(r)

def reconstruct_tree(MapD, MapS, U, V, n_nodes):
#    T = [hashlib.new('sha1')]*n_nodes
    T = ['']*n_nodes
    for key,val in MapD.items():
        (node,src,id) = val;
        if id == len(U):
            T[node] = key
        else:
            copy_subtree(T,U,V,key,node,src,id)
    for key,val in MapS.items():
        (node,src,id) = val
        T[key] = T[src]
    return T


#test_str0 = "abcdefgh"
#test_str0 = "Hello Muddah"
test_str0 = "Hello Muddah. Hello Fadduh. Here I am at camp Granada"
#test_str0 = "Hello Mother. Hello Father. Here I am at camp Granada"

#test_str1 = "abcdacdc"
#test_str1 = "Hello Fadduh"
test_str1 = "Hello Mother. Hello Father. Here I am at camp Granada"
#test_str1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
print(len(test_str1))
chunk_size = 1
tree0 = create_merkle_tree(test_str0, chunk_size)
print_tree(tree0)
tree1 = create_merkle_tree(test_str1, chunk_size)
print_tree(tree1)

(Md0, Ms0) = find_distinct_subtrees(tree0, 0)
tree1 = create_merkle_tree(test_str1, chunk_size)
#print_tree(tree1)
#print(len(Md0))

(Md1, Ms1) = find_distinct_subtrees(tree1, 1)
print("Distinct("+str(len(Md1))+"): " + str(Md1) + str('\n'))
print("Shared("+str(len(Ms1))+"): " + str(Ms1) + str('\n'))
(Md1, Ms1) = deduplication(tree1, [Md0])
print("Distinct("+str(len(Md1))+"): " + str(Md1) + str('\n'))
print("Shared("+str(len(Ms1))+"): " + str(Ms1) + str('\n'))

#print("Reconstructed tree")
#tree = reconstruct_tree(Md, Ms, [Md0], [Ms0], len(tree1))
#print_tree(tree)
