
#include "common_custom.h"

#include <queue>
#include <utility>
#include <vector>

class pair_compare {
public:
    bool operator() (std::pair<int, float> x, std::pair<int, float> y) {
        // need a max heap, so this operation should be greater-than
        return std::get<1>(x) > std::get<1>(y);
    }
};

typedef std::priority_queue<std::pair<int, float>,
                            std::vector<std::pair<int, float>>,
                            pair_compare>
        heap_t;

extern "C" {

void* heap_alloc() {
    return (void*) new heap_t();
}
void heap_free(void* heap) {
    delete (heap_t*)heap;
}

bool heap_empty(void* heap) {
    return ((heap_t*)heap)->empty();
}

int heap_remove(void *heap) {
    int value = std::get<0>(((heap_t*)heap)->top());
    ((heap_t*)heap)->pop();
    return value;
}
void heap_insert(void *heap, int value, float w) {
    ((heap_t*)heap)->push({value, w});
}

}
