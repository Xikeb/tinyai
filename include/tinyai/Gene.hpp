#pragma once

namespace neat {
    struct Gene {
		unsigned int innovation_num = -1;
		unsigned int from_node = -1;
		unsigned int to_node = -1;
		double weight = 0.0;
		bool enabled = true;
	};
}