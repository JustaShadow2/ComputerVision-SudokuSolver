class Solution(object):
    def solveSudoku(self, board):
        self.board = board
        self.solve()

    def findEmpty(self):
        for row in range(9):
            for col in range(9):
                if self.board[row][col] == '.':
                    return (row, col)
        return None, None

    def solve(self):
        row, col = self.findEmpty()
        if row == None and col == None:
            return True
        for i in range(1, 10): 
            if self.check(row, col, i): # if the number is valid for the current row and column then we can place it in the board and move on to the next empty cell 
                self.board[row][col] = str(i) # place the number in the board 
                if self.solve(): # if no more empty cells are found
                    return True
                self.board[row][col] = '.' # backtrack and try a different number if the current number doesn't work
        return False

    def check(self, row, col, num):
        for x in range(9): # check if the number is already in the current row
            if self.board[row][x] == str(num) and x != col: # if the number is already in the current row and it's not the current column
                return False
            if self.board[x][col] == str(num) and x != row: # if the number is already in the current column and it's not the current row
                return False
        box_x = row // 3 # find the current box [x]
        box_y = col // 3 # find the current box [y]
        for i in range(box_x * 3, box_x * 3 + 3): # 
            for j in range(box_y * 3, box_y * 3 + 3): # 
                if self.board[i][j] == str(num) and (i, j) != (row, col): # if the number is already in the current box and it's not the current cell 
                    return False
        return True
    
    def printBoard(self):
        for row in self.board:
            print(row)

# # test
# board = [["5","3",".",".","7",".",".",".","."],
#         ["6",".",".","1","9","5",".",".","."],
#         [".","9","8",".",".",".",".","6","."],
#         ["8",".",".",".","6",".",".",".","3"],
#         ["4",".",".","8",".","3",".",".","1"],
#         ["7",".",".",".","2",".",".",".","6"],
#         [".","6",".",".",".",".","2","8","."],
#         [".",".",".","4","1","9",".",".","5"],
#         [".",".",".",".","8",".",".","7","9"]]

# sudoku = Solution()
# sudoku.solveSudoku(board)
# sudoku.printBoard()