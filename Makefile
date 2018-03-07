legion_install_dir=/usr/local/
ldflags=-L${legion_install_dir}/lib64/ -llegion -lrealm -lpthread -ldl -rdynamic \
			  -lrt 
cflags=-std=c++14 -Wall -I${legion_install_dir}/include -I. -g
CC=g++

examples=examples/fib examples/hello examples/region examples/2d examples/stream

all: ${examples}

%: %.cc equites.h 
	${CC} ${cflags} -o $@ $< ${ldflags}

fib: 
	equites.h

clean: 
	rm ${examples}
