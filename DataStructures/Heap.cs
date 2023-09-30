using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class Heap
    {
        private List<int> _heap;
        private Func<int, int, bool> _comparisionFunc;
        public int _length;

        public Heap(List<int> array, Func<int, int, bool> comparisionFunc){
            this._comparisionFunc= comparisionFunc;
            this._heap = BuildHeap(array);            
            this._length= this._heap.Count;
        }
        public int Peek(){
            return this._heap[0];
        }
        public int Remove(){
            this.Swap(0, _heap.Count-1,_heap);
            int valueToRemove = _heap[_heap.Count-1];
            this._heap.RemoveAt(_heap.Count-1);
            this._length -=1;
            this.SiftDown(0,_heap.Count -1,_heap);
            return valueToRemove;
        }
        public void Insert(int value){
            this._heap.Add(value);
            this._length+=1;
            this.SiftUp(_heap.Count-1);
        }

        private void SiftUp(int currentIdx){
            int parentIdx =(currentIdx-1)/2;
            while(currentIdx > 0){
                if(this._comparisionFunc(this._heap[currentIdx], _heap[parentIdx])){
                    Swap(currentIdx,parentIdx,_heap);
                    currentIdx=parentIdx;
                    parentIdx=(currentIdx-1)/2;                    
                }else{return;}
                
            }
        }
        private List<int> BuildHeap(List<int> array){
            int firstParentIdx = (array.Count -2)/2;
            for(int currentIdx=firstParentIdx; currentIdx>=0; currentIdx--){
                this.SiftDown(currentIdx, array.Count-1, array);
            }
            return array;
        }

        private void SiftDown(int currentIdx, int endIdx, List<int> heap){
            int childOneIdx = currentIdx*2+1;
            while(childOneIdx <= endIdx){
                int childTwoIdx = currentIdx *2+2 <= endIdx ? currentIdx*2+2:-1;

                int idxToSwap;
                if(childTwoIdx != -1){
                    if(this._comparisionFunc(heap[childTwoIdx],heap[childOneIdx])){
                        idxToSwap=childTwoIdx;
                    }else idxToSwap= childOneIdx;
                }else{
                    idxToSwap= childOneIdx;
                }
                if(this._comparisionFunc(heap[idxToSwap],heap[currentIdx])){
                        Swap(currentIdx, idxToSwap, heap);
                        currentIdx=idxToSwap;
                        childOneIdx=currentIdx*2+1;
                }else{return;}

            }           
        }

        private void Swap(int currentIdx, int idxToSwap, List<int> heap)
        {
            int temp = heap[currentIdx];
            heap[currentIdx] = heap[idxToSwap];
            heap[idxToSwap] = temp; 
        }
        public static bool MAX_HEAP_FUNC(int a, int b){
            return a>b;
        }
        public static bool MIN_HEAP_FUNC(int a, int b){
            return a<b;
        }
    }
}