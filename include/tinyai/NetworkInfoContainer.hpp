#pragma once

namespace neat {
    struct NetworkInfoContainer {
		unsigned int input_size;
		unsigned int bias_size;
		unsigned int output_size;
		unsigned int functional_nodes;
		bool recurrent;
	};
}