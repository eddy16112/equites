legion_install_dir=/usr/local/
ldflags=-L${legion_install_dir}/lib64/ -llegion -lrealm -lpthread -ldl -rdynamic \
			  -lrt 
cflags=-std=c++14 -Wall -I${legion_install_dir}/include -I.. -g
CC=g++

examples=fib hello region 2d stream inc doublewrite

all: ${examples}

%: %.cc ../equites.h 
	${CC} ${cflags} -o $@ $< ${ldflags}

clean: 
	rm ${examples}
