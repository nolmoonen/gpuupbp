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

#ifndef KERNEL_SARRAY_CUH
#define KERNEL_SARRAY_CUH

#include "defs.cuh"

/// Static array.
template <typename T, unsigned int N>
class SArray {
  public:
    __forceinline__ __device__ SArray() : size_(0), start(elements){};

    __forceinline__ __device__ unsigned int size() const { return size_; }

    /// Is array empty?
    __forceinline__ __device__ bool empty() const { return size_ == 0; }

    /// Returns false iff array is full.
    /// Do not throw error here since it is important to know which
    /// type caused the overflow.
    __forceinline__ __device__ bool push_back(const T& elem)
    {
        if (size_ == N) return false;
        start[size_++] = elem;
        return true;
    }

    /// Remove element from start
    __forceinline__ __device__ void pop_front()
    {
        GPU_ASSERT(size_ != 0);
        ++start;
        --size_;
    }

    /// Clear array
    __forceinline__ __device__ void clear()
    {
        size_ = 0;
        start = elements;
    }

    __forceinline__ __device__ T& operator[](unsigned int i)
    {
        GPU_ASSERT(size_ > i);
        return start[i];
    }

    __forceinline__ __device__ const T& operator[](unsigned int i) const
    {
        GPU_ASSERT(size_ > i);
        return start[i];
    }

    __forceinline__ __device__ T& front()
    {
        GPU_ASSERT(size_ > 0);
        return start[0];
    }

  public:
    T elements[N];
    T* start;
    unsigned int size_;
};

#endif // KERNEL_SARRAY_CUH
