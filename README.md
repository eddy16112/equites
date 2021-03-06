equites
=======

High level, type-safe interface to legion. 

```cpp
task(void, print){
  printf("hello world\n");
}

int main(int argc, char** argv){
  start(print, argc, argv);
}
```

See `examples/` for example usage. Running `make` will build all examples.

## Features
- Type-safe legion calls
- Type-safe region semantics
- Automatic management of region arguments
- Regent-like syntax
- C++ range based for loop

## TODO
- Partitioning
- Field spaces
- Inline mapping
- Reductions
- Unstructured regions
- Rewrite Apps (Stencil)
- Contiguous iterator optimizations (look at lambdas)
- Templated interfaces (e.g. IndexSpaceT)
- Indexed task launches
