__author__ = 'Deepak Sharma'
#Implementation of minimax algorithm with and without using alpha beta pruning
# Author: Deepak Sharma, RIT, SEP, 2016

class game:
    """
    This class represent the problem definition,
    provides goal test and successor function utilities
    """
    __slots__ = 'board', 'player1_marker', 'player2_marker'

    def __init__(self, pl1_Marker,pl2_marker):
        """
        initialize the game
        :param pl1_Marker: Marker type of the player 1
        :param pl2_marker: Marker type of the player 2
        :return:
        """
        self.player1_marker=pl1_Marker
        self.player2_marker=pl2_marker
        self.board=[[" "," "," "],[" "," "," "],[" "," "," "]]
        #self.board=[["X","X","O"],["0","O","X"],[" "," "," "]]

    def successor_function(self):
        """
        provides the list of all available actions
        :return: List containing actions
        """
        actions=[]
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col]==" ":
                    actions.append((row,col))
        return actions

    def termination_test(self):
        """
        Check for the winning configuration on the tic tac toe board
        :return: return the player number  of the winning player if game terminates,
        else return None
        """
        Vpos=[(1,0),(1,1),(1,2)]
        Hpos=[(0,1),(1,1),(2,1)]
        empty=True
        for pos in Vpos:
            if self.board[pos[0]][pos[1]] !=" ":
                if self.board[pos[0]][pos[1]] == self.board[pos[0]-1][pos[1]] == self.board[pos[0]+1][pos[1]]:
                    return 1  if self.board[pos[0]][pos[1]]==self.player1_marker else 2
            if self.board[pos[0]][pos[1]]==" " or self.board[pos[0]-1][pos[1]]==" " or self.board[pos[0]+1][pos[1]]==" ":
                    empty=False

        for pos in Hpos:
            if self.board[pos[0]][pos[1]] !=" ":
                if self.board[pos[0]][pos[1]] == self.board[pos[0]][pos[1]-1] == self.board[pos[0]][pos[1]+1]:
                    return 1  if self.board[pos[0]][pos[1]]==self.player1_marker else 2
        if self.board[1][1] !=" ":
            if self.board[1][1] == self.board[0][0] == self.board[2][2]:

                    return 1  if self.board[1][1]==self.player1_marker else 2

            if self.board[1][1] == self.board[0][2] == self.board[2][0]:
                    return 1  if self.board[1][1]==self.player1_marker else 2

        if empty:
            return "empty"
        return None
    def show_move_locations(self):
        """
        Show coordinates of the boards
        :return:
        """

        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                print("("+str(row+1)+","+str(col+1)+")",end=" ")
            print()

    def showBoard(self):
        """
        Show board to the human player
        :return:
        """
        print()
        for row in self.board:
            for element in row:
                if element!=" ":
                    print(element, end=" ")
                else:
                    print("_", end=" ")
            print()

    def is_Valid_move(self,xCor,yCor):
        """
        Check the validity of a move
        :param xCor: X coordinate
        :param yCor: Y coordinate
        :return: True if move is valid
        """

        if xCor>=0 and xCor<=2 and yCor>=0 and yCor<=2:
            return self.board[xCor][yCor]==" "

    def place_move (self,xCor,yCor,player_num):
        """
        update the board status with provided move
        :param xCor: X coordinate of the move
        :param yCor: Y coordinate of the move
        :param player_num: player number [1,2]
        :return: None
        :pre The move should be valid move, us is_valid_move for checking the validity of a move
        """
        if player_num==1:
            self.board[xCor][yCor]=self.player1_marker
        elif player_num==2:
            self.board[xCor][yCor]=self.player2_marker


    def remove_move(self,xCor,yCor):
        """
        Undo the provided move
        :param xCor: X coordinate
        :param yCor: Y coordinate
        :return: None
        Pre: The method should only be used by computer player
        while executing min max algorithm, Human player class should have
        No access to the method
        """
        self.board[xCor][yCor]=" "

class HumanPlayer:
    """
    The class represents the human player, provide move function
    for making a move in the game
    """
    __slots__= 'game','playerNumber'

    def __init__(self,game,playerNumber):
        """
        intialize the players
        :param game: Game object
        :param playerNumber:
        :return:
        """
        self.playerNumber=playerNumber
        self.game=game


    def move(self):
        """
        enable the user to make a move
        :return:
        """
        print("It is your turn")
        print("Enter location coma separated: row-number(1-3),col-number(1-3) Example: 1,1")
        while(True):
            try:
                location=input().split(sep=",")
                if self.game.is_Valid_move(int(location[0])-1,int(location[1])-1):
                    self.game.place_move(int(location[0])-1,int(location[1])-1,self.playerNumber)
                    break
                else:
                    print("Not a valid move Sir, try again!!!")
            except:
                print("Not a valid input Sir, try again!!!")
class computerPlayer:
    """
    This class represents the computer player, It implements the
    MiniMax algorithm for deciding the move for computer player
    """
    __slots__= 'game','playerNumber'

    def __init__(self,game,playerNumber):
        """
        Initialize the object of computer player
        :param game: Game object
        :param playerNumber:
        :return: none
        """
        self.playerNumber=playerNumber
        self.game=game
    def __Max_Value__(self):
        """
        Max value method of the minimax algorithm
        :return:
        """
        node_number=0
        game_status=self.game.termination_test()
        if game_status=="empty":
            return (0,None,node_number+1)
        if game_status!=None :
            return (1,None,node_number+1) if game_status==self.playerNumber else (-1,None,node_number+1)
        utility={}
        #print("Max")
        #print(self.game.successor_function())
        for action in self.game.successor_function():
            self.game.place_move(action[0],action[1],self.playerNumber)
            value,location,nodes=self.__Min_Value__()
            #print(nodes,"Inside Max, my Child gave me")
            node_number+=nodes
            if value in utility.keys():
               utility[value].append(action)
            else:
                utility[value]=[action]
            self.game.remove_move(action[0],action[1])
        #if  utility[max(utility.keys())] !=None:
        #print("max")
        #print(utility)
        return max(utility.keys()),utility[max(utility.keys())][0],node_number+1

    def __Min_Value__(self):
        """
        min method of the minimax algorithm
        :return:
        """
        node_number=0
        game_status=self.game.termination_test()
        if game_status=="empty":
            return (0,None,node_number+1)
        if game_status!=None:
            return (1,None,node_number+1) if game_status==self.playerNumber else (-1,None,node_number+1)
        utility={}
        #print("min")
        #print(self.game.successor_function())
        child_nodes=0
        for action in self.game.successor_function():
            self.game.place_move(action[0],action[1],self.playerNumber-1)
            value,location,nodes=self.__Max_Value__()
            node_number+=nodes
            #print(nodes,"Inside Min, my Child gave me")
            if value in utility.keys():
               utility[value].append(action)
            else:
                utility[value]=[action]
            self.game.remove_move(action[0],action[1])
        #if utility[min(utility.keys())] !=None:
        #print("Min")
        #print(utility)
        return min(utility.keys()),utility[min(utility.keys())][0],node_number+1


    def __Min_Value__alpha_beta(self,alpha,beta):
        """
        Min method of the MiniMax algorithm with alpha beta pruning
        :param alpha:
        :param beta:
        :return:selected Move,utility value and number of search nodes generated
        """
        node_number=0
        utility={}#OrderedDict()
        game_status=self.game.termination_test()
        if game_status=="empty":
            return (0,None,node_number+1)
        if game_status!=None:
            return (1,None,node_number+1) if game_status==self.playerNumber else (-1,None,node_number+1)

        for action in self.game.successor_function():
            self.game.place_move(action[0],action[1],self.playerNumber-1)
            value,location,nodes=self.__Max_Value__alpha_beta(alpha,beta)
            node_number+=nodes
            if value in utility.keys():
               utility[value].append(action)
            else:
                utility[value]=[action]
            self.game.remove_move(action[0],action[1])
            if value<=alpha:
                return value,utility[value],node_number+1
            beta=min(beta,value)
        if utility[min(utility.keys())] !=None:
            return min(utility.keys()),utility[min(utility.keys())][0],node_number+1

    def __Max_Value__alpha_beta(self,alpha,beta):
        """
        Max method of the minimax algorithm with alpah beta pruning
        :param alpha:
        :param beta:
        :return: selected Move,utility value and number of search nodes generated
        """
        node_number=0
        utility={}#OrderedDict()
        game_status=self.game.termination_test()
        if game_status=="empty":
            return (0,None,node_number+1)
        if game_status!=None :
            return (1,None,node_number+1) if game_status==self.playerNumber else (-1,None,node_number+1)

        for action in self.game.successor_function():
            self.game.place_move(action[0],action[1],self.playerNumber)
            value,location,nodes=self.__Min_Value__alpha_beta(alpha,beta)
            node_number+=nodes
            if value in utility.keys():
               utility[value].append(action)
            else:
                utility[value]=[action]
            self.game.remove_move(action[0],action[1])
            if value>=beta:
                return value,utility[value][0],node_number+1
            alpha=max(alpha,value)

        if  utility[max(utility.keys())] !=None:
            return max(utility.keys()),utility[max(utility.keys())][0],node_number+1

    def _Min_Max_decision(self):
        """
        intiate minimax algorithm with alpha beta pruning
        :return:
        """
        return self.__Max_Value__()

    def _Min_Max_alpha_beta_decision(self):
        """
        intiate minimax algorithm with alpha beta pruning
        :return: Move, utility value and number of search nodes
        """
        return self.__Max_Value__alpha_beta(-2,2)

    def move(self):
        """
        Initiate the minimax algorithm for generating a move

        :return:  None
        """
        value,move_Location,search_nodes=self._Min_Max_decision();

        #print(value)
        #print(move_Location)
        print("Minimax algorithm has generated: "+str(search_nodes)+" search nodes for deciding a move")
        print("Computer has decided move:" + str(move_Location))
        value,move_Location,search_nodes=self._Min_Max_alpha_beta_decision();
        print("Minimax algorithm  with alpha beta pruning has generated: "+str(search_nodes)+" search nodes for deciding a move")
        print("Computer has decided move:" + str(move_Location))

        self.game.place_move(move_Location[0],move_Location[1],self.playerNumber)

class Game_Controller:
    """
    This class organize the game
    """
    __slot__='new_game','player!','player2'

    def __init__(self):

        self.new_game=game("X","O")
        self.player1=HumanPlayer(self.new_game,1)
        self.player2=computerPlayer(self.new_game,2)

    def start_game(self):
        print("Welcome to the Tic tac toe game")
        print("your piece looks like: X")
        print("Computer's piece looks like: O")
        print("Move locations coordinates are:")
        self.new_game.show_move_locations()
        while(True):
            #self.new_game.showBoard()
            #print(self.new_game.termination_test())
            if self.new_game.termination_test()==None:
                self.player1.move()
                self.new_game.showBoard()
            if self.new_game.termination_test()==None:
                self.player2.move()
                self.new_game.showBoard()
            else:
                if self.new_game.termination_test()!="Equal":
                    player= "Human " if self.new_game.termination_test()==1 else "Computer"
                    print(player, " Won the game" )
                else:
                    print("Draw game")
                print("*********Game Over********")
                break

def main():
    """
    Main method, create object of the Game_controller and initiate the game

    :return:
    """
    lets_Have_a_game=Game_Controller()
    lets_Have_a_game.start_game()

main()
