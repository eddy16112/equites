legion_install_dir=/home/wwu12/legion-install
ldflags=-L${legion_install_dir}/lib64/ -llegion -lrealm -lpthread -ldl -rdynamic \
			  -lrt 
cflags=-std=c++14 -Wall -I${legion_install_dir}/include -I.. -g
CC=g++

examples=hello_world fib index_tasks saxpy_index saxpy 2d  multiple_partitions
#examples=vpsc_example

all: ${examples}

%: %.cc ../equites.h 
	${CC} ${cflags} -o $@ $< ${ldflags}

clean: 
	rm ${examples}
