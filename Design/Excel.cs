using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    631. Design Excel Sum Formula
    https://leetcode.com/problems/design-excel-sum-formula/

    Approach 1: Topological Sort
    Time Complexity:
        Set: takes O((r∗c)^2 time. Here, r and c refer to the number of rows and columns in the current Excel Form. There can be a maximum of O(r∗c) formulas for an Excel Form with r rows and c columns. For each formula, r∗c time will be needed to find the dependent nodes. 
            Thus, in the worst case, a total of O((r∗c)^2) will be needed.
        Sum: takes O((r∗c)^2+2∗r∗c∗l) time. Here, l refers to the number of elements in the the list of strings used for obtaining the cells required for the current sum. 
            In the worst case, the expansion of each such element requires O(r∗c) time, leading to O(l∗r∗c) time for expanding l such elements. After doing the expansion, calculate_sum itself requires O(l∗r∗c) time for traversing over the required elements for obtaining the sum. 
            After this, we need to update all the dependent cells, which requires the use of set which itself requires O((r∗c)^2) time.
        Get: takes O(1) time.
    Space Complexity: The space required will be O((r∗c)^2) in the worst case. O(r∗c) space will be required for the Excel Form itself. For each cell in this form, the cells list can contain O(r∗c) cells.
    */
    public class Excel
    {
        private Formula[,] formulas;
        private Stack<int[]> stack = new Stack<int[]>();

        private class Formula
        {
            public Dictionary<string, int> Cells { get; private set; }
            public int Value { get; private set; }

            public Formula(Dictionary<string, int> cells, int value)
            {
                Value = value;
                Cells = cells;
            }
        }

        public Excel(int height, char width)
        {
            formulas = new Formula[height, (width - 'A') + 1];
        }

        public int Get(int row, char column)
        {
            if (formulas[row - 1, column - 'A'] == null)
                return 0;
            return formulas[row - 1, column - 'A'].Value;
        }

        public void Set(int row, char column, int value)
        {
            formulas[row - 1, column - 'A'] = new Formula(new Dictionary<string, int>(), value);
            TopologicalSort(row - 1, column - 'A');
            ExecuteStack();
        }

        public int Sum(int row, char column, string[] strs)
        {
            var cells = Convert(strs);
            int sum = CalculateSum(row - 1, column - 'A', cells);
            Set(row, column, sum);
            formulas[row - 1, column - 'A'] = new Formula(cells, sum);
            return sum;
        }

        private void TopologicalSort(int row, int column)
        {
            for (int i = 0; i < formulas.GetLength(0); i++)
            {
                for (int j = 0; j < formulas.GetLength(1); j++)
                {
                    if (formulas[i, j] != null && formulas[i, j].Cells.ContainsKey("" + (char)('A' + column) + (row + 1)))
                    {
                        TopologicalSort(i, j);
                    }
                }
            }
            stack.Push(new int[] { row, column });
        }

        private void ExecuteStack()
        {
            while (stack.Count > 0)
            {
                int[] top = stack.Pop();
                if (formulas[top[0], top[1]].Cells.Count > 0)
                    CalculateSum(top[0], top[1], formulas[top[0], top[1]].Cells);
            }
        }

        private Dictionary<string, int> Convert(string[] strs)
        {
            var result = new Dictionary<string, int>();
            foreach (string str in strs)
            {
                if (str.IndexOf(":") < 0)
                    result[str] = result.GetValueOrDefault(str, 0) + 1;
                else
                {
                    string[] cells = str.Split(':');
                    int startIndex = int.Parse(cells[0].Substring(1));
                    int endIndex = int.Parse(cells[1].Substring(1));
                    char startChar = cells[0][0];
                    char endChar = cells[1][0];
                    for (int i = startIndex; i <= endIndex; i++)
                    {
                        for (char j = startChar; j <= endChar; j++)
                        {
                            result["" + j + i] = result.GetValueOrDefault("" + j + i, 0) + 1;
                        }
                    }
                }
            }
            return result;
        }

        private int CalculateSum(int row, int column, Dictionary<string, int> cells)
        {
            int sum = 0;
            foreach (string key in cells.Keys)
            {
                int x = int.Parse(key.Substring(1)) - 1;
                int y = key[0] - 'A';
                sum += (formulas[x, y] != null ? formulas[x, y].Value : 0) * cells[key];
            }
            formulas[row, column] = new Formula(cells, sum);
            return sum;
        }
    }
    /**
    * Your Excel object will be instantiated and called as such:
    * Excel obj = new Excel(height, width);
    * obj.Set(row,column,val);
    * int param_2 = obj.Get(row,column);
    * int param_3 = obj.Sum(row,column,numbers);
    */
}