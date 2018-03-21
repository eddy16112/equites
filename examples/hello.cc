#include <equites.h>
using namespace equites;

task(void, hello){
  printf("hello world\n");
}

int main(int argc, char** argv){
  start(hello, argc, argv);
}

