using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class SetOps
    {
        //https://www.algoexpert.io/questions/powerset
        public static List<List<int>> GeneratePowerSet(List<int> array){


            //1.Iterative - T:O(n*2^n) | O(n*2^n)
            List<List<int>> powersets = GeneratePowerSetIterative(array );

            //2.Recursive - T:O(n*2^n) | O(n*2^n)
            powersets = GeneratePowerSetRecursive(array, array.Count-1);
            return powersets;
        }

        private static List<List<int>> GeneratePowerSetRecursive(List<int> array, int idx)
        {
            if(idx < 0){
                List<List<int>> emptySet = new List<List<int>>();
                emptySet.Add(new List<int>());
                return emptySet;
            }

            int element = array[idx];
            List<List<int>> subsets = GeneratePowerSetRecursive(array, idx-1);
            int length = subsets.Count;
            for(int i=0; i< length; i++){
                List<int> currentSubset = new List<int>(subsets[i]);
                currentSubset.Add(element);
                subsets.Add(currentSubset);
            }
            return subsets;
        }

        private static List<List<int>> GeneratePowerSetIterative(List<int> array)
        {

            List<List<int>> subSets = new List<List<int>>();
            subSets.Add(new List<int>());

            foreach(var element in array){
                int length = subSets.Count;

                for(int i=0; i< length; i++){
                    List<int> currentSubset = new List<int>(subSets[i]);
                    currentSubset.Add(element);
                    subSets.Add(currentSubset);

                }
            }
            return subSets;
            
        }
    }
}