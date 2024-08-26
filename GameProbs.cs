using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Security.Cryptography.X509Certificates;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace AlgoDSPlay
{
    public class GameProbs
    {
        //https://www.algoexpert.io/questions/reveal-minesweeper
        public static string[][] RevealMinesweeper(string[][] board, int row, int col){
            //O(w * h) time | O(w * h) space - where w is the width of the board, and h is the height of the board
            if(board[row][col] == MINE){
                board[row][col]= CLOSE;
                return board;
            }
            List<CellLocation> neighbors = GetNeighbors(board, row, col);

            int adjacentMinesCount =0;
            foreach(var neighbor in neighbors){
                if(board[neighbor.Row][neighbor.Col].Equals(MINE))
                    adjacentMinesCount++;
            }

            if(adjacentMinesCount >0 ){
                board[row][col]= adjacentMinesCount.ToString();
            }else{
                board[row][col]="0";
                foreach(var neighbor in neighbors){
                    if(board[neighbor.Row][neighbor.Col].Equals(NOMINE))
                        RevealMinesweeper(board, neighbor.Row, neighbor.Col);
                }

            }
            return board;
        }

        private static List<CellLocation> GetNeighbors(string[][] board, int row, int col)
        {
            int[,] directions = new int[8,2]{
                {0,1},{0,-1}, //Same row - left and right
                {1,-1},{1,0},{1,1}, //next row - bottom-left, down and right and bottom-right diagonal
                {-1,-1},{-1,0},{-1,1} //previous row - top-left, up and top-right diagonal
            };

            List<CellLocation> neighbors = new List<CellLocation>();
            for(int i=0; i< directions.GetLength(0); i++){
                int newRow = row+directions[i,0];
                int newCol = col+directions[i,1];

                if(0 <= newRow && newRow < board.Length && 0<= newCol && newCol < board[0].Length )
                    neighbors.Add(new CellLocation(newRow, newCol));
                
            }
            return neighbors;

        }

        public static string MINE ="M";
        public static string CLOSE ="X";
        public static string NOMINE ="H";
        public class CellLocation{
            public int Row;
            public int Col;

            public CellLocation(int row, int col){
                this.Row = row;
                this.Col = col;
            }
        }
    }
}