using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.DataStructures
{
    public class Heap<T>
    {
        private List<T> _heap;
        private Func<T, T, bool> _comparisionFunc;

        public Heap(List<T> array, Func<T, T, bool> comparisionFunc){
            this._comparisionFunc= comparisionFunc;
            this._heap = BuildHeap(array);            
        }
        public T Peek(){
            return this._heap[0];
        }
        public bool IsEmpty(){
            return _heap.Count ==00;
        }
        public int Count{
            get{ return _heap.Count;}
        }
        public T Remove(){
            this.Swap(0, _heap.Count-1,_heap);
            T valueToRemove = _heap[_heap.Count-1];
            this._heap.RemoveAt(_heap.Count-1);
            this.SiftDown(0,_heap.Count -1,_heap);
            return valueToRemove;
        }
        public void Insert(T value){
            this._heap.Add(value);
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
        private List<T> BuildHeap(List<T> array){
            int firstParentIdx = (array.Count -2)/2;
            for(int currentIdx=firstParentIdx; currentIdx>=0; currentIdx--){
                this.SiftDown(currentIdx, array.Count-1, array);
            }
            return array;
        }

        private void SiftDown(int currentIdx, int endIdx, List<T> heap){
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

        private void Swap(int currentIdx, int idxToSwap, List<T> heap)
        {
            T temp = heap[currentIdx];
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