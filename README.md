equites
=======

High level, type-safe interface to legion. 

```
task(void, print){
  printf("hello world\n");
}

int main(int argc, char** argv){
  start(print, argc, argv);
  return 0;
}
```

See `examples/` for example usage. Running `make` will build all examples.

# Features
- Type safe legion calls
- Type safe region semantic
- Automatic management of region arguments
- Regent-like syntax

# TODO
- Partitioning
- Field spaces
- Inline mapping
- Iterators

