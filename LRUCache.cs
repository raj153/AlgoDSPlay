using System.Collections.Generic;
using AlgoDSPlay.DataStructures;

namespace AlgoDSPlay{

    public class LRUCache<T1, T2>{

        public Dictionary<T1,DoublyLinkedListNode<T1, T2>> _cache = new Dictionary<T1, DoublyLinkedListNode<T1, T2>>();
        public int _maxSize;
        public int _currentSize =0;
        public DoublyLinkedList<T1, T2> listOfMostRecent = new DoublyLinkedList<T1, T2>();

        public LRUCache(int maxSize){
            this._maxSize = maxSize>1? _maxSize:1;
        }

        //T:O(1): S:O(1)
        public void InsertKeyValuePair(T1 key, T2 value){

            if(!_cache.ContainsKey(key)){
                if(_currentSize == _maxSize){
                    EvictLeaseRecent();
                }else _currentSize++;
                _cache.Add(key,new DoublyLinkedListNode<T1, T2>(key,value));
            }else{ ReplaceKey(key,value);}
            
            UpdateMostRecent(_cache[key]);

        }
        //T:O(1) : S:O(1)
        public LRUResult<T2> GetValueFromKey(T1 key){
            if(!_cache.ContainsKey(key)) return new LRUResult<T2>(false, default(T2));
            UpdateMostRecent(_cache[key]);
            return new LRUResult<T2>(true, _cache[key].Value);
        }
        //T:O(1) : S:O(1)
        public T1 GetMostRecentKey(){
            if(listOfMostRecent.Head == null) return default(T1);

            return listOfMostRecent.Head.Key;
        }

        public void EvictLeaseRecent(){
            T1 keyToRemove = listOfMostRecent.Tail.Key;
            listOfMostRecent.RemoveTail();
            _cache.Remove(keyToRemove);
        }
        public void UpdateMostRecent(DoublyLinkedListNode<T1, T2> node){
            listOfMostRecent.SetHeadTo(node);
        }
        public void ReplaceKey(T1 key, T2 value){
            if(!this._cache.ContainsKey(key)) return;
            _cache[key].Value = value;
        }
     
    }
    public class LRUResult<T>{
        public bool Found;
        public T Value;

        public LRUResult(bool found, T value){
            Found= found;
            Value = value;
        }
    }
}