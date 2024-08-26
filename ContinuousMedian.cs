using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay
{
    //https://www.algoexpert.io/questions/continuous-median
    public class ContinuousMedian
    {
        public Heap<int> _maxHeap; //Lower/First Half
        public Heap<int> _minHeap; //Greater/second half
        public double _median=0;

        public ContinuousMedian(){
            this._minHeap = new Heap<int>(new List<int>(), Heap<int>.MIN_HEAP_FUNC);
            this._maxHeap = new Heap<int>(new List<int>(), Heap<int>.MAX_HEAP_FUNC);
            this._median=0;
        }

        //T:O(logN) | S:O(n)
        public void Insert(int number){
            if(_maxHeap.IsEmpty() || number < _maxHeap.Peek())
                _maxHeap.Insert(number);
            else
                _minHeap.Insert(number);
            
            this.RebalanceHeaps();
            this.UpdateMedian();
        }

        private void RebalanceHeaps()
        {
            if(_maxHeap.Count - _minHeap.Count == 2){
                _minHeap.Insert(_maxHeap.Remove());                
            }else if(_minHeap.Count-_maxHeap.Count == 2){
                _maxHeap.Insert(_minHeap.Remove());
            }
        }
        private void UpdateMedian(){
            if(_maxHeap.Count == _minHeap.Count){
                this._median = ((double)_maxHeap.Peek()+(double)_minHeap.Peek())/2;
            }else if(_maxHeap.Count > _minHeap.Count)
                this._median = _maxHeap.Peek();
            else this._median = _minHeap.Peek();
        }
        public double GetMedian(){
            return this._median;
        }
    }
}