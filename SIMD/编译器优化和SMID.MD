## 指针的索引尽量使用size_t

size_t在64位系统上uint64_t

size_t在32位系统上uint32_t

所以不需要用movslq从32位符号拓展到64位，更高效，而且能处理数组大小超过INT_MAX的情况，推荐使用size_t处理数组大小和索引

## 造成new/delete的容器，内存分配在堆上的容器

存储在堆上妨碍优化，存储在栈上有利优化

堆上容器vector, map, set, string, function, any, unique_ptr, shared_ptr, weak_ptr

栈上容器array, bitset, glm::vec, string_view, pair, tuple, optional, variant

## 写简单的代码，复杂的代码编译器会放弃优化

或者使用constexpr，强制编译器在编译器求值

constexpr无法使用非constexpr的容器vector,map,set

但是C++20好像可以

## 内联函数





## 告诉编译器别怕指针别名

可以使用__restrict的关键字，告诉编译器这些指针不会重叠，所以编译器可以直接优化成\*c=\*b;。不然编译器会害怕func(&a,&b,&b)这种情况的出现，编译器总是区域保守，不会优化出快的结果，只会保证正确的结果。

```c++
void func(int *__restrict a,int *__restrict b,int *__restrict c){
	*c=*a;
	*c=*b;
}
```

__restrict只需要加在非const指针上即可，const是只读变量。结论就是所有非const指针都应该声明该关键字

```
void func(int const* a,int const* b,int *__restrict c){
	*c=*a;
	*c=*b;
}
```

