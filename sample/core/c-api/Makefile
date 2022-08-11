PYTHON_DEPS:= \
		-I/opt/python3/include/python3.8 \
		-L/opt/python3/lib \
		-lpython3.8

all: test_simple test_multiply

prepare:
	mkdir -p out

simple: prepare
	gcc -o out/simple string/simple.c -g \
		$(PYTHON_DEPS)

test_simple: simple
	./out/simple

multiply: prepare
	gcc -o out/multiply -g \
		file/multiply/multiply.c \
		$(PYTHON_DEPS)

test_multiply: multiply
	cp file/multiply/multiply.py out
	PYTHONPATH=$PYTHONPATH:out ./out/multiply multiply multiply 3 2

clean:
	rm -rf out

