using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2353. Design a Food Rating System
    https://leetcode.com/problems/design-a-food-rating-system/description/

    */
    public class FoodRatings
    {
        /*        
        Approach 1: Hash Maps and Priority Queue (HMPQ)
        Complexity Analysis
        Here, n is the initial size of the foods array, and let, m be the number of calls made to changeRating and highestRated methods.
        •	Time complexity: O(nlogn+mlog(n+m))
            o	Initialization:
                o	We iterate over all foods elements and insert them into appropriate hash maps and priority queues. Inserting a value into the hash map takes constant time, but, inserting a value into the priority queue will take logarithmic time.
                o	Thus, for n elements, the total time taken will be O(nlogn) time.
            o	changeRating(food, newRating) method:
                o	Updating the rating in the hash map will take constant time.
                o	But, in the worst case, the priority queue can contain (n+m) elements, and inserting an element into the priority queue will take O(log(n+m)) time.
                o	Thus, for m insertions, the total time taken will be O(mlog(n+m)) time.
            o	highestRated(cuisine) method:
                o	Getting the cuisine name from the hash map and the top element of the priority queue are both constant time operations.
                o	But, we might also remove some elements from the priority queue. Each removal operation will take O(log(n+m)) time.
                o	Each element is permanently unused after it is removed, i.e. they are removed at most once, so, for all highestRated method calls we may remove at most m elements.
                o	Thus, the total time taken for all calls will be O(mlog(n+m)) time.
        •	Space complexity: O(n+m)
            o	In foodRatingMap, and foodCuisineMap we will store all n elements, thus, they both will take O(n) space.
            o	In cuisineFoodMap we might insert (n+m) elements, thus, it will take O(n+m) space.
        */
        public class Food : IComparable<Food>
        {
            // Store the food's rating.
            public int FoodRating { get; set; }
            // Store the food's name.
            public string FoodName { get; set; }

            public Food(int foodRating, string foodName)
            {
                FoodRating = foodRating;
                FoodName = foodName;
            }

            // Implement the compareTo method for comparison
            public int CompareTo(Food other)
            {
                // If food ratings are the same, sort based on their names (lexicographically smaller name food will be on top)
                if (FoodRating == other.FoodRating)
                {
                    return FoodName.CompareTo(other.FoodName);
                }
                // Sort based on food rating (bigger rating food will be on top)
                return -1 * FoodRating.CompareTo(other.FoodRating);
            }
        }

        public class FoodRatingsHMPQ
        {
            // Map food with its rating.
            private Dictionary<string, int> foodRatingMap;
            // Map food with the cuisine it belongs to.
            private Dictionary<string, string> foodCuisineMap;

            // Store all food of a cuisine in a priority queue (to sort them on ratings/name)
            // Priority queue element -> Food: (foodRating, foodName)
            private Dictionary<string, PriorityQueue<Food, Food>> cuisineFoodMap;

            public FoodRatingsHMPQ(string[] foods, string[] cuisines, int[] ratings)
            {
                foodRatingMap = new Dictionary<string, int>();
                foodCuisineMap = new Dictionary<string, string>();
                cuisineFoodMap = new Dictionary<string, PriorityQueue<Food, Food>>();

                for (int i = 0; i < foods.Length; ++i)
                {
                    // Store 'rating' and 'cuisine' of the current 'food' in 'foodRatingMap' and 'foodCuisineMap' maps.
                    foodRatingMap[foods[i]] = ratings[i];
                    foodCuisineMap[foods[i]] = cuisines[i];
                    // Insert the '(rating, name)' element into the current cuisine's priority queue.
                    if (!cuisineFoodMap.ContainsKey(cuisines[i]))
                    {
                        cuisineFoodMap[cuisines[i]] = new PriorityQueue<Food, Food>();
                    }
                    var food = new Food(ratings[i], foods[i]);
                    cuisineFoodMap[cuisines[i]].Enqueue(food, food);
                }
            }

            public void ChangeRating(string food, int newRating)
            {
                // Update food's rating in the 'foodRating' map.
                foodRatingMap[food] = newRating;
                // Insert the '(new food rating, food name)' element into the respective cuisine's priority queue.
                string cuisineName = foodCuisineMap[food];
                var foodee = new Food(newRating, food);
                cuisineFoodMap[cuisineName].Enqueue(foodee, foodee);
            }

            public string HighestRated(string cuisine)
            {
                // Get the highest rated 'food' of 'cuisine'.
                Food highestRated = cuisineFoodMap[cuisine].Peek();

                // If the latest rating of 'food' doesn't match the 'rating' on which it was sorted in the priority queue,
                // then we discard this element from the priority queue.
                while (foodRatingMap[highestRated.FoodName] != highestRated.FoodRating)
                {
                    cuisineFoodMap[cuisine].Dequeue();
                    highestRated = cuisineFoodMap[cuisine].Peek();
                }

                // Return the name of the highest-rated 'food' of 'cuisine'.
                return highestRated.FoodName;
            }
        }

        /*                
        Approach 2: Hash Maps and Sorted Set (HMSS)

        Complexity Analysis
        Here, n is the initial size of the foods array, and let, m be the number of calls made to changeRating and highestRated methods.
        •	Time complexity: O((n+m)logn)
            o	Initialization:
                o	We iterate over all foods elements and insert them into appropriate hash maps and sorted sets. Inserting a value into the hash map takes constant time, but, inserting a value into the sorted set will take logarithmic time.
                o	Thus, for n elements, the total time taken will be O(nlogn) time.
            o	changeRating(food, newRating) method:
                o	Updating the rating in the hash map will take constant time.
                o	But, the sorted set will have n elements, and inserting and deleting an element in it will take O(logn) time.
                o	Thus, for m insertions, the total time taken will be O(mlogn) time.
            o	highestRated(cuisine) method:
                o	Getting the cuisine name from the hash map is a constant time operation.
                o	The sorted set will have n elements, in C++ and Java, getting the min element will take logn time but in Python, it will take O(1) time.
                o	Thus, the total time taken for m calls in C++ and Java will be O(mlogn) and in Python will be O(m).
        •	Space complexity: O(n)
            o	In foodRatingMap, foodCuisineMap, and cuisineFoodMap we will store n elements.
            o	Thus, overall it will take O(n) space.
        */

        public class FoodRatingsHMSS
        {
            // Map food with its rating.
            private Dictionary<string, int> foodRatingMap = new Dictionary<string, int>();
            // Map food with cuisine it belongs to.
            private Dictionary<string, string> foodCuisineMap = new Dictionary<string, string>();

            // Store all food of cuisine in set (to sort them on ratings/name)
            // Set element -> Pair: (-1 * foodRating, foodName)
            private Dictionary<string, SortedSet<(int rating, string foodName)>> cuisineFoodMap = new Dictionary<string, SortedSet<(int rating, string foodName)>>();

            public FoodRatingsHMSS(string[] foods, string[] cuisines, int[] ratings)
            {
                for (int i = 0; i < foods.Length; ++i)
                {
                    // Store 'rating' and 'cuisine' of current 'food' in 'foodRatingMap' and 'foodCuisineMap' maps.
                    foodRatingMap[foods[i]] = ratings[i];
                    foodCuisineMap[foods[i]] = cuisines[i];

                    // Insert the '(-1 * rating, name)' element in the current cuisine's set.
                    if (!cuisineFoodMap.ContainsKey(cuisines[i]))
                    {
                        cuisineFoodMap[cuisines[i]] = new SortedSet<(int rating, string foodName)>(Comparer<(int rating, string foodName)>.Create((a, b) =>
                        {
                            int compareByRating = a.rating.CompareTo(b.rating);
                            if (compareByRating == 0)
                            {
                                // If ratings are equal, compare by food name (in ascending order).
                                return a.foodName.CompareTo(b.foodName);
                            }
                            return compareByRating;
                        }));
                    }
                    cuisineFoodMap[cuisines[i]].Add((-ratings[i], foods[i])); 
                }
            }

            public void ChangeRating(string food, int newRating)
            {
                // Fetch cuisine name for food.
                string cuisineName = foodCuisineMap[food];

                // Find and delete the element from the respective cuisine's set.
                var cuisineSet = cuisineFoodMap[cuisineName];
                var oldElement = (-foodRatingMap[food], food);
                cuisineSet.Remove(oldElement);

                // Update food's rating in 'foodRating' map.
                foodRatingMap[food] = newRating;
                // Insert the '(-1 * new rating, name)' element in the respective cuisine's set.
                cuisineSet.Add((-newRating, food));
            }

            public string HighestRated(string cuisine)
            {
                var highestRated = cuisineFoodMap[cuisine].Min;
                // Return name of the highest rated 'food' of 'cuisine'.
                return highestRated.foodName;
            }
        }

    }
    /**
 * Your FoodRatings object will be instantiated and called as such:
 * FoodRatings obj = new FoodRatings(foods, cuisines, ratings);
 * obj.ChangeRating(food,newRating);
 * string param_2 = obj.HighestRated(cuisine);
 */
}