#include <equites.h>
using namespace equites;

task(void, print){
  printf("hello world\n");
}

int main(int argc, char** argv){
  start(print, argc, argv);
}

