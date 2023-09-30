using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class MinHeapForTDArray
    {
        List<List<int>> heap = new List<List<int>>();
  
        public MinHeapForTDArray(List<List<int>> array){
            heap = BuildHeap(array);
        }

        //T:O(n) | S:O(1)
        private List<List<int>> BuildHeap(List<List<int>> array)
        {
            int firstParentIdx = (array.Count-2)/2;
            for(int currentIdx=firstParentIdx+1; currentIdx >=0; currentIdx--){
                SiftDown(currentIdx, array.Count-1, array);
            }
            return array;
        }
        public bool IsEmpty(){
            return heap.Count == 0;
        }

        //T:O(log(n)) | S:O(1)
        private void SiftDown(int currentIdx, int endIdx, List<List<int>> array)
        {
            int childOneIdx = currentIdx*2+1;
            while(childOneIdx <= endIdx){
                int childTwoIdx = currentIdx *2+2 <= endIdx? currentIdx*2+2:-1;
                int idxToSwap;
                if(childTwoIdx != -1 && 
                    array[childTwoIdx][1] < array[childOneIdx][1]){
                        idxToSwap = childTwoIdx;
                    }else {idxToSwap = childOneIdx;}

                if(array[idxToSwap][1] < array[currentIdx][1]){
                    Swap(currentIdx,idxToSwap);
                    currentIdx = idxToSwap;
                    childOneIdx = currentIdx*2+1;

                }
                else{
                    return;
                }

            }

        }
        //T:O(log(n) | S:O(1)
        private void SiftUp(int currentIdx){
            int parentIdx= (currentIdx-1)/2;
            while(currentIdx >0 && heap[currentIdx][1] < heap[parentIdx][1]){
                Swap(currentIdx,parentIdx);
                currentIdx = parentIdx;
                parentIdx = (currentIdx-1)/2;
            }
        }
        public List<int> Peek(){
            return heap[0];
        }
        
        public List<int> Remove(){
            if(IsEmpty()) return null;

            Swap(0, heap.Count-1);
            List<int> valueToRemove = heap[heap.Count-1];
            heap.RemoveAt(heap.Count-1);
            SiftDown(0, heap.Count-1, heap);
            return valueToRemove;

        }
        public void Insert(List<int> value){
            heap.Add(value);
            SiftUp(heap.Count-1);
        }
        private void Swap(int idx1, int idx2)
        {
        
            List<int> temp = heap[idx1];
            heap[idx1] = heap[idx2];
            heap[idx2] = temp;

            
        }
    }
}