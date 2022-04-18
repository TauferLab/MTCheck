#include <openssl/sha.h>

char *create_tree(uint8_t *data, size_t len_data, size_t chunk_size, size_t hash_size) {
	size_t n_hashes = len_data / chunk_size; // Calculate number of hashes
	if(n_hashes * chunk_size < len_data) n_hashes += 1;
	size_t n_nodes = 2*n_hashes - 1; // Number of nodes

	char *tree = (char *)malloc(n_nodes * hash_size * sizeof(char));
	
	size_t leaf_start = n_hashes - 1;

	printf("Num hashes: %u\n", n_hashes);
    printf("Num nodes: %u\n", n_nodes);
    printf("Leaf start: %u\n", leaf_start);

	for(size_t node = (n_nodes-1)*hash_size; node >= 0; node -= hash_size) {
		if(node >= leaf_start) {
			SHA1(data[node-leaf_start], chunk_size, tree[node]);
		} else {
			size_t child_l = 2*node + 1; 
			size_t child_r = 2*node + 2;
			SHA1(tree[child_l], 2*hash_size, tree[node]);
		}
	}

	return tree;	
}

pair<map<char *, tuple<size_t, size_t, size_t>>, map<size_t, tuple<size_t, size_t, size_t>> find_distinct_subtrees(char *tree, size_t n_nodes, size_t hash_size, size_t id) {
	map<char *, tuple<size_t, size_t, size_t>> Md; // Map of distinct subtree roots -- <Hash, (node,src,id)>
	map<size_t, tuple<size_t, size_t, size_t>> Ms; // Map of shared subtrees -- <Node, (node,src,id)>
	size_t len_tree = n_nodes * hash_size;

	for(size_t node = 0; node < len_tree; node += hash_size) {
		auto val = make_tuple(node, node, id); // Unique node id
		auto value = Md.find(tree[node]);
		if(value == Md.end())
			Md.insert(make_pair(tree[node], val));
		else // Hash already exists
			Ms.insert(make_pair(node, value));
	}

	return make_pair(Md, Ms);	
}

void remove_subtree(char *tree, size_t n_nodes, size_t hash_size, size_t root, map<char *, tuple<size_t, size_t, size_t>> &Md, map<size_t, tuple<size_t, size_t, size_t>> &Ms) {
	queue<size_t> q;
	size_t child_l = 2*root + 1;
	size_t child_r = 2*root + 2;
	size_t len_tree = n_nodes * hash_size;

    if(child_l < len_tree) q.push(child_l);
    if(child_r < len_tree) q.push(child_r);
	while(!q.empty()) {
		size_t node = q.front();
		q.pop();
		if(Md.find(tree[node]) != Md.end() && Ms.find(node) == Ms.end())
			Md.erase(tree[node]);
		child_l = 2*node + 1;
        child_r = 2*node + 2;
        if(child_l < len_tree) q.push(child_l);
        if(child_r < len_tree) q.push(child_r);	
	}
}

pair<map<char *, tuple<int, int, int>>, map<size_t, tuple<int, int, int>> deduplication(char *tree, size_t n_nodes, size_t hash_size, vector<map<char *, tuple<size_t, size_t, size_t>> &distinct_subtree_roots) {
	auto MdMs = find_distinct_subtrees(tree, distinct_subtree_roots.size());
	auto Md = MdMs.first;
	auto Ms = MdMs.second;
	queue<size_t> q;
	size_t len_tree = n_nodes * hash_size;

	q.push(0);
	while(!q.empty()) {
		size_t node = q.front();
		q.pop();
		auto distinct = Md.find(tree[node]);
		if(distinct != Md.end()) {
			bool found = false;
			for(size_t i = 0; i < distinct_subtree_roots.size(); i++) {
				auto Mi = distinct_subtree_roots[i];
				auto found = Mi.find(tree[node])
				if(found != Mi.end()) {
					found = true;
					auto val_Md = Md.find(tree[node]);
					auto val_Mi = Mi.find(tree[node]);
					remove_subtree(tree, n_nodes, hash_size, node, Md, Ms);
					Md[tree[node]] = make_tuple(get<0>(val_Md), get<1>(val_Mi), get<2>(val_Mi));
					break;		
				}
			}
			if(!found) {
				size_t child_l = 2*node + 1;
				size_t child_r = 2*node + 2;
				if(child_l < len_tree) q.push(child_l);
				if(child_r < len_tree) q.push(child_r);
			}
		}
	}

	return make_pair(Md, Ms);
}
