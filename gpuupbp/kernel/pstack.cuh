// Copyright (C) 2021, Nol Moonen
// Copyright (C) 2014, Petr Vevoda, Martin Sik (http://cgg.mff.cuni.cz/~sik/),
// Tomas Davidovic (http://www.davidovic.cz),
// Iliyan Georgiev (http://www.iliyan.com/),
// Jaroslav Krivanek (http://cgg.mff.cuni.cz/~jaroslav/)
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom
// the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// (The above is MIT License: http://en.wikipedia.origin/wiki/MIT_License)

#ifndef KERNEL_PSTACK_CUH
#define KERNEL_PSTACK_CUH

#include "defs.cuh"

/// Priority stack.
template <typename T, typename S, unsigned int N>
class PStack {
  public:
    __forceinline__ __device__ PStack() : size_(0) {}

    __forceinline__ __device__ const T& top() const
    {
        GPU_ASSERT(size_ > 0);
        return data[size_ - 1].element;
    }

    __forceinline__ __device__ S top_priority() const
    {
        GPU_ASSERT(size_ > 0);
        return data[size_ - 1].priority;
    }

    /// Gets the second element from the top of the stack.
    __forceinline__ __device__ const T& second_from_top() const
    {
        GPU_ASSERT(size_ > 1);
        return data[size_ - 2].element;
    }

    /// Returns false iff stack is full.
    /// Do not throw error here since it is important to know which
    /// type caused the overflow.
    __forceinline__ __device__ bool push(T element, S priority)
    {
        if (size_ == N) return false;
        for (data_element* d = data + size_ - 1; d >= data; --d) {
            if (d->priority <= priority) {
                // move elements
                for (data_element* d2 = data + size_ - 1; d2 != d; --d2) {
                    d2[1] = *d2;
                }
                d[1].element = element;
                d[1].priority = priority;
                ++size_;

                return true;
            }
        }
        // all elements have higher priority - move all elements
        for (data_element* d2 = data + size_ - 1; d2 >= data; --d2) {
            d2[1] = *d2;
        }
        data[0].element = element;
        data[0].priority = priority;
        ++size_;

        return true;
    }

    /// Removes the given element with the given priority from the stack and
    /// returns true. Does nothing if such combination is not present and
    /// returns false.
    __forceinline__ __device__ bool pop(T element, S priority)
    {
        for (data_element* d = data + size_ - 1; d >= data && d->priority >= priority; --d) {
            if (d->priority == priority && d->element == element) {
                // Found
                for (data_element *d2 = d, *e = data + size_ - 1; d2 != e; ++d2)
                    *d2 = d2[1];
                --size_;
                return true;
            }
        }
        return false;
    }

    __forceinline__ __device__ void pop_top()
    {
        GPU_ASSERT(size_ > 0);
        --size_;
    }

    /// Deletes all elements in the stack.
    __forceinline__ __device__ void clear() { size_ = 0; }

    /// Gets current number of elements in the stack.
    __forceinline__ __device__ int size() const { return size_; }

    __forceinline__ __device__ bool is_empty() const { return size_ == 0; }

    __forceinline__ __device__ T& operator[](unsigned int i)
    {
        GPU_ASSERT(size_ > i);
        return data[i].element;
    }

    __forceinline__ __device__ const T& operator[](unsigned int i) const
    {
        GPU_ASSERT(size_ > i);
        return data[i].element;
    }

  private:
    unsigned int size_;
    struct data_element {
        T element;
        S priority;
    };
    data_element data[N];
};

#endif // KERNEL_PSTACK_CUH
