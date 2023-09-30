using System.Threading.Tasks.Dataflow;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class NQueens
    {
        //https://www.algoexpert.io/questions/non-attacking-queens
        public int NonAttackingQueens(int n){
            //1.Naive - T: N*N! - Impossible or long running
            
            //2.Lower bound: T: O(n!) | S:O(n)
            int[] columnPlacements = new int[n];

            int result= GetNumberOfNonAttackingQueenPlacements(0, columnPlacements,n);

            //3.Backtracking - Upper bound: T: O(n!) | S:O(n) 
            HashSet<int> blockingColumns = new HashSet<int>();
            HashSet<int> blockingUpDiagonals = new HashSet<int>();
            HashSet<int> blockingDownDiagonals = new HashSet<int>();

            result = GetNumberOfNonAttackingQueenPlacementsWithBackTracking(0, blockingColumns, blockingUpDiagonals, blockingDownDiagonals,n);
            return result;
        }

        private int GetNumberOfNonAttackingQueenPlacementsWithBackTracking(int row, HashSet<int> blockingColumns, HashSet<int> blockingUpDiagonals, HashSet<int> blockingDownDiagonals, int boardSize)
        {
            if(row == boardSize) return 1;

            int validPlacements =0;

            for(int col=0; col< boardSize; col++){
                if(isNonAttackingPlacement(row, col, blockingColumns, blockingUpDiagonals, blockingDownDiagonals)){
                    PlaceQueen(row, col, blockingColumns, blockingDownDiagonals, blockingUpDiagonals);

                    validPlacements += GetNumberOfNonAttackingQueenPlacementsWithBackTracking(row+1, blockingColumns, blockingDownDiagonals, blockingUpDiagonals, boardSize);
                    RemoveQueen(row, col, blockingColumns, blockingDownDiagonals, blockingUpDiagonals);
                }
            }

            return validPlacements;
            
        }

        private void RemoveQueen(int row, int col, HashSet<int> blockingColumns, HashSet<int> blockingDownDiagonals, HashSet<int> blockingUpDiagonals)
        {
            blockingColumns.Remove(col);
            blockingDownDiagonals.Remove(row-col);
            blockingUpDiagonals.Remove(row+col);
        }

        private void PlaceQueen(int row, int col, HashSet<int> blockingColumns, HashSet<int> blockingDownDiagonals, HashSet<int> blockingUpDiagonals)
        {
            blockingColumns.Add(col);
            blockingDownDiagonals.Add(row-col);
            blockingUpDiagonals.Add(row+col);
        }

        private bool isNonAttackingPlacement(int row, int col, HashSet<int> blockingColumns, HashSet<int> blockingUpDiagonals, HashSet<int> blockingDownDiagonals)
        {
            if(blockingColumns.Contains(col))
                return false;
            if(blockingDownDiagonals.Contains(row-col))
                return false;
            if(blockingUpDiagonals.Contains(row+col))
                return false;
            
            return true;
        }

        private int GetNumberOfNonAttackingQueenPlacements(int row, int[] columnPlacements, int boardSize)
        {
            if(row==boardSize) return 1;

            int validPlacements =0;

            for(int col=0; col<boardSize; ++col){
                if(isNonAttackingPlacement(row, col, columnPlacements)){
                    columnPlacements[row]=col;
                    validPlacements += GetNumberOfNonAttackingQueenPlacements(row+1, columnPlacements, boardSize);
                }
            }

            return validPlacements;
        }

        private bool isNonAttackingPlacement(int row, int col, int[] columnPlacements)
        {
            for(int previousRow=0; previousRow < row; ++previousRow){
                int columnToCheck = columnPlacements[previousRow];
                bool sameColumn = col == columnToCheck;
                bool sameDiagonal = Math.Abs(columnToCheck-col) == Math.Abs(previousRow-row);

                if(sameColumn || sameDiagonal)
                    return false;
            }
            return true;
            
        }
    }
}