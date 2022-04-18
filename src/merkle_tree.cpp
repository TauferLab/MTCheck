#include <openssl/sha.h>

char *create_tree(char *data, size_t chunk_size, size_t hash_size) {
	size_t len_data = (size_t)((sizeof(data) / sizeof(data[0])));
	size_t n_hashes = len_data / chunk_size; // Calculate number of hashes
	if(n_hashes * chunk_size < len_data) n_hashes += 1;
	size_t n_nodes = 2*n_hashes - 1; // Number of nodes

	char *tree = (char *)malloc(n_nodes * hash_size * sizeof(char));
	
	size_t leaf_start = n_hashes - 1;

	printf("Num hashes: %u\n", n_hashes);
    printf("Num nodes: %u\n", n_nodes);
    printf("Leaf start: %u\n", leaf_start);

	for(size_t node = n_nodes-1; node >= 0; node -= hash_size) {
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

pair<map<char *, tuple<size_t, size_t, size_t>>, map<size_t, tuple<size_t, size_t, size_t>> find_distinct_subtrees(char *tree, size_t id) {
	size_t n_nodes = (size_t)((sizeof(tree) / sizeof(tree[0]))); // this is n_nodes * hash_size -- want n_nodes
	map<char *, tuple<size_t, size_t, size_t>> Md // Map of distinct subtrees roots -- <Hash, (node,src,id)>
	map<size_t, tuple<size_t, size_t, size_t>> Ms // Map of shared subtrees -- <Node, (node,src,id)>

	for(size_t node = 0; node < n_nodes; node++) {
		auto val = make_tuple(node, node, id); // Unique node id
		auto value = Md.find(tree[node]);
		if(value == Md.end())
			Md.insert(make_pair(tree[node], val));
		else // Hash already exists
			Ms.insert(make_pair(node, value));
	}

	return make_pair(Md, Ms);	
}


pair<map<char *, tuple<int, int, int>>, map<size_t, tuple<int, int, int>> deduplication(char *tree, /*array of maps*/) {
	auto MdMs = find_distinct_subtrees(tree, /*id*/);
	auto Md = MdMs.first;
	auto Ms = MdMs.second;
}
