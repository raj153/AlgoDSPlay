using System;
using System.Collections.Generic;
using System.Formats.Asn1;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;

namespace AlgoDSPlay
{
    public class BinarySearchProbs
    {
        //https://www.algoexpert.io/questions/shifted-binary-search
        public static int ShiftedBinarySearch(int[] array, int target){

            //1.Recursive
            //T:O(log(n)) | S:(n)
            int result = ShiftedBinarySearchRec(array, target, 0, array.Length-1);

            //2.Iterative
            //T:O(log(n)) | S:(1)
            result = ShiftedBinarySearchIterative(array, target, 0, array.Length-1);

            return result;
        }

        private static int ShiftedBinarySearchIterative(int[] array, int target, int left, int right)
        {
            while(left <=right){
                int middle = (left+right)/2;
                int potentialMatch = array[middle];
                int leftNum = array[left];
                int rightNum = array[right];

                if(target == potentialMatch) return middle;
                else if(leftNum <= potentialMatch){
                    if(target< potentialMatch && target >= leftNum){
                        right = middle-1;
                    }
                    else left = middle+1;
                }else{
                    if(target > potentialMatch && target <= rightNum){
                        left = middle+1;                        
                    }else right = middle-1;
                }
            }
            return -1;
        }

        private static int ShiftedBinarySearchRec(int[] array, int target, int left, int right)
        {
            if(left > right) return -1;

            int middle = (left+right)/2;
            int leftNum= array[left];
            int rightNum= array[right];
            int potentialMatch = array[middle];
            if(target == potentialMatch){
                return middle;
            }
            else if(left <= potentialMatch){
                if(target < potentialMatch && target >= leftNum ){
                    return ShiftedBinarySearchRec(array, target, left, middle-1);
                }else{
                    return ShiftedBinarySearchRec(array, target, middle+1, right);
                }
            }else {
                if(target > potentialMatch && target <= rightNum){
                    return ShiftedBinarySearchRec(array, target, middle+1, rightNum);
                }else{
                    return ShiftedBinarySearchRec(array, target, left, middle-1);
                }
            }
            
            
        }
        public static int[] SearchForRange(int[] array, int target){
            int[] finalRange = new int[]{-1,-1};
            
            //1. Recursion
            //T:O(log(n)) | S:O(log(n))
            AlteredBinarySearchRec(array, target, 0, array.Length-1, finalRange, true);
            AlteredBinarySearchRec(array, target, 0, array.Length-1, finalRange, false);

            //2. Iterative with constant space
            //T:O(log(n)) | S:O(1)
            AlteredBinarySearchIterative(array, target, 0, array.Length-1, finalRange, true);
            AlteredBinarySearchIterative(array, target, 0, array.Length-1, finalRange, false);
            return finalRange;
        }

        private static void AlteredBinarySearchIterative(int[] array, int target, int left, int right, int[] finalRange, bool goLeft)
        {
            while( left <= right){
                int mid =(left+right)/2;
                if(array[mid] < target){
                    left = mid+1;
                }else if(array[mid] > target){
                    right = mid-1;
                }else{
                    if(goLeft){
                        if(mid ==0 || array[mid-1] !=target){
                            finalRange[0]=mid;
                            return;
                        }else{
                            right = mid-1;
                        }
                    }else{
                        if(mid == array.Length -1 || array[mid+1] != target){
                            finalRange[1] = mid;
                        }else{
                            left = mid+1;
                        }
                    }
                }
            }
            
        }

        private static void AlteredBinarySearchRec(int[] array, int target, int left, int right, int[] finalRange, bool goLeft)
        {
            if(left > right) return;

            int mid =(left+right)/2;

            if(array[mid] < target){
                AlteredBinarySearchRec(array, target, mid+1, right, finalRange, goLeft );
            }else if(array[mid]> target){
                AlteredBinarySearchRec(array, target, left, mid-1, finalRange,  goLeft);
            }else{
                if(goLeft){
                    if( mid ==0 || array[mid-1] != target){
                        finalRange[0]=mid;
                    }else{
                        AlteredBinarySearchRec(array, target, left, mid-1, finalRange, goLeft);
                    }
                }else{
                    if(mid == array.Length -1 || array[mid+1] != target){
                        finalRange[1] = mid;                        
                    }else{
                        AlteredBinarySearchRec(array, target, mid+1, right, finalRange, goLeft);
                    }
                }
            }
        }
    }
}