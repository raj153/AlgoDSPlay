using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1603. Design Parking System
    https://leetcode.com/problems/design-parking-system/

    Complexity Analysis
    Let N be the number of function calls.
    •	Time complexity: O(N).
        o	In the constructor, we create one array of constant size, empty. Hence, the time complexity will be O(1).
        o	In the addCar function, we check if the number of empty slots for a particular type of car is greater than 0. This is done in constant time.
    Hence, the overall time complexity will be O(N) since there are N function-calls.
    •	Space complexity: O(1).
        o	In the constructor, we create one array of constant size, empty. Hence, the space complexity will be O(1).
        o	In the addCar function, we do not use any extra space. Hence, the space complexity will be O(1).
    Hence, the overall space complexity will be O(1).

    */
    public class ParkingSystem
    {

        // Number of empty slots for each type of car
        int[] empty;

        public ParkingSystem(int big, int medium, int small)
        {
            this.empty = new int[] { big, medium, small };
        }

        public bool AddCar(int carType)
        {

            // If space is available, allocate and return True
            if (this.empty[carType - 1] > 0)
            {
                this.empty[carType - 1]--;
                return true;
            }

            // Else, return False
            return false;
        }
    }    /**
 * Your ParkingSystem object will be instantiated and called as such:
 * ParkingSystem obj = new ParkingSystem(big, medium, small);
 * bool param_1 = obj.AddCar(carType);
 */
}