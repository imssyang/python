all:
	./main

str:
	g++ -g -o str str.cpp -I/opt/python3/include/python3.8 -L/opt/python3/lib -lpython3.8

long:
	g++ -g -o long long.cpp -I/opt/python3/include/python3.8 -L/opt/python3/lib -lpython3.8

func:
	gcc -g -o func func.cpp -I/opt/python3/include/python3.8 -L/opt/python3/lib -lpython3.8 -lstdc++

cmd:
	gcc -g -o cmd cmd.cpp -I/opt/python3/include/python3.8 -L/opt/python3/lib -lpython3.8 -lstdc++

clean:
	rm -rf str long func cmd __pycache__ pycmd/__pycache__

