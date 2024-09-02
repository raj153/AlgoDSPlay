using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    353. Design Snake Game
    https://leetcode.com/problems/design-snake-game/description/

    Complexity Analysis
    Let W represent the width of the grid and H represent the height of the grid. Also, let N represent the number of food items in the list.
    •	Time Complexity:
        o	The time complexity of the move function is O(1).
        o	The time taken to calculate bites_itself is constant since we are using a dictionary to search for the element.
        o	The time taken to add and remove an element from the queue is also constant.
    •	Space Complexity:
        o	The space complexity is O(W×H+N)
        o	O(N) is used by the food data structure.
        o	O(W×H) is used by the snake and the snake_set data structures. At most, we can have snake that occupies all the cells of the grid as explained in the beginning of the article.


    */

    public class SnakeGame
    {
        private Dictionary<(int, int), bool> snakeMap;
        private LinkedList<(int, int)> snake;
        private int[][] food;
        private int foodIndex;
        private int width;
        private int height;

        /**
         * Initialize your data structure here.
         *
         * @param width - screen width
         * @param height - screen height
         * @param food - A list of food positions E.g food = [[1,1], [1,0]] means the first food is
         *     positioned at [1,1], the second is at [1,0].
         */
        public SnakeGame(int width, int height, int[][] food)
        {
            this.width = width;
            this.height = height;
            this.food = food;
            this.snakeMap = new Dictionary<(int, int), bool>();
            this.snakeMap[(0, 0)] = true; // initially at [0][0]
            this.snake = new LinkedList<(int, int)>();
            this.snake.AddLast((0, 0));
        }

        /**
         * Moves the snake.
         *
         * @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
         * @return The game's score after the move. Return -1 if game over. Game over when snake crosses
         *     the screen boundary or bites its body.
         */
        public int Move(string direction)
        {
            var snakeCell = this.snake.First.Value;
            int newHeadRow = snakeCell.Item1;
            int newHeadColumn = snakeCell.Item2;

            switch (direction)
            {
                case "U":
                    newHeadRow--;
                    break;
                case "D":
                    newHeadRow++;
                    break;
                case "L":
                    newHeadColumn--;
                    break;
                case "R":
                    newHeadColumn++;
                    break;
            }

            var newHead = (newHeadRow, newHeadColumn);
            var currentTail = this.snake.Last.Value;

            // Boundary conditions.
            bool crossesBoundary1 = newHeadRow < 0 || newHeadRow >= this.height;
            bool crossesBoundary2 = newHeadColumn < 0 || newHeadColumn >= this.width;

            // Checking if the snake bites itself.
            bool bitesItself = this.snakeMap.ContainsKey(newHead) && !(newHead.Item1 == currentTail.Item1 && newHead.Item2 == currentTail.Item2);

            // If any of the terminal conditions are satisfied, then we exit with code -1.
            if (crossesBoundary1 || crossesBoundary2 || bitesItself)
            {
                return -1;
            }

            // If there's an available food item and it is on the cell occupied by the snake after the move,
            // eat it.
            if ((this.foodIndex < this.food.Length)
                && (this.food[this.foodIndex][0] == newHeadRow)
                && (this.food[this.foodIndex][1] == newHeadColumn))
            {
                this.foodIndex++;
            }
            else
            {
                this.snake.RemoveLast();
                this.snakeMap.Remove(currentTail);
            }

            // A new head always gets added
            this.snake.AddFirst(newHead);

            // Also add the head to the set
            this.snakeMap[newHead] = true;

            return this.snake.Count - 1;
        }
    }


    /**
 * Your SnakeGame object will be instantiated and called as such:
 * SnakeGame obj = new SnakeGame(width, height, food);
 * int param_1 = obj.Move(direction);
 */
}