legion_install_dir=/home/wwu12/legion-install
ldflags=-L${legion_install_dir}/lib64/ -llegion -lrealm -lpthread -ldl -rdynamic \
			  -lrt 
cflags=-std=c++14 -Wall -I${legion_install_dir}/include -I.. -g
CC=g++

examples= saxpy_inline hello_world fib index_tasks saxpy_index saxpy 2d multiple_partitions saxpy_subregion sub_task inline
#examples=vpsc_example 

all: ${examples}

legion_simplified.o: ../legion_simplified.cc ../legion_simplified.h
	${CC} ${cflags} -o $@ -c $< 

%: %.cc legion_simplified.o ../legion_simplified.h 
	${CC} ${cflags} -o $@ $< legion_simplified.o ${ldflags}

clean: 
	rm ${examples}
