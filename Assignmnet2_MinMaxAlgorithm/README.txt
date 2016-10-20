run the program using below command

python3 ttt.py


It will show below message:

	Welcome to the Tic tac to game
	your piece looks like: X
	Computer's piece looks like: O
	Move locations coordinates are:
	(1,1) (1,2) (1,3) 
	(2,1) (2,2) (2,3) 
	(3,1) (3,2) (3,3) 
	It is your turn
	Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1


Enter any corrdinate from command prompt in the format of x,y


After making a move it will give below output:

	Minimax algorithm has generated: 59705 search nodes for deciding a move
	Computer has decided move:(1, 1)
	Minimax algorithm  with alpha beta proning has generated: 2338 search nodes for deciding a move
	Computer has decided move:(1, 1)

First line of the output says number of search nodes generated using min max algorithm

Second line tells the move decided

their lines tell number of search nodes generated using min max with alph beta pruning algorithm

forth line says move decided by alpha beta proning algorithm


***************************************************************** output ************************************************

Welcome to the Tic tac toe game
your piece looks like: X
Computer's piece looks like: O
Move locations coordinates are:
(1,1) (1,2) (1,3) 
(2,1) (2,2) (2,3) 
(3,1) (3,2) (3,3) 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
1,1

X _ _ 
_ _ _ 
_ _ _ 
Minimax algorithm has generated: 59705search nodes for deciding a move
Computer has decided move:(1, 1)
Minimax algorithm  with alpha beta proning has generated: 2338search nodes for deciding a move
Computer has decided move:(1, 1)

X _ _ 
_ O _ 
_ _ _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
2,2
Not a valid move Sir, try again!!!
3,1

X _ _ 
_ O _ 
X _ _ 
Minimax algorithm has generated: 927search nodes for deciding a move
Computer has decided move:(1, 0)
Minimax algorithm  with alpha beta proning has generated: 189search nodes for deciding a move
Computer has decided move:(1, 0)

X _ _ 
O O _ 
X _ _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
3,2

X _ _ 
O O _ 
X X _ 
Minimax algorithm has generated: 38search nodes for deciding a move
Computer has decided move:(1, 2)
Minimax algorithm  with alpha beta proning has generated: 29search nodes for deciding a move
Computer has decided move:(1, 2)

X _ _ 
O O O 
X X _ 
Computer  Won the game
*********Game Over********


************************************************************ output ******************************************************

Welcome to the Tic tac toe game
your piece looks like: X
Computer's piece looks like: O
Move locations coordinates are:
(1,1) (1,2) (1,3) 
(2,1) (2,2) (2,3) 
(3,1) (3,2) (3,3) 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
2,2

_ _ _ 
_ X _ 
_ _ _ 
Minimax algorithm has generated: 55505search nodes for deciding a move
Computer has decided move:(0, 0)
Minimax algorithm  with alpha beta proning has generated: 2316search nodes for deciding a move
Computer has decided move:(0, 0)

O _ _ 
_ X _ 
_ _ _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
3,1

O _ _ 
_ X _ 
X _ _ 
Minimax algorithm has generated: 933search nodes for deciding a move
Computer has decided move:(0, 2)
Minimax algorithm  with alpha beta proning has generated: 116search nodes for deciding a move
Computer has decided move:(0, 2)

O _ O 
_ X _ 
X _ _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
1,2

O X O 
_ X _ 
X _ _ 
Minimax algorithm has generated: 51search nodes for deciding a move
Computer has decided move:(2, 1)
Minimax algorithm  with alpha beta proning has generated: 39search nodes for deciding a move
Computer has decided move:(2, 1)

O X O 
_ X _ 
X O _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
2,1

O X O 
X X _ 
X O _ 
Minimax algorithm has generated: 5search nodes for deciding a move
Computer has decided move:(1, 2)
Minimax algorithm  with alpha beta proning has generated: 5search nodes for deciding a move
Computer has decided move:(1, 2)

O X O 
X X O 
X O _ 
It is your turn
Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1
3,3

O X O 
X X O 
X O X 
Computer  Won the game
*********Game Over********







