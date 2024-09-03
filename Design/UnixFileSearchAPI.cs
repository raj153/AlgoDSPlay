using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    public class UnixFileSearchAPI
    {



        private void Test()
        {
            var params_ = new SearchParams
            {
                Extension = "xml",
                MinSize = 2,
                MaxSize = 100
            };

            var xmlFile = new File
            {
                Content = Encoding.UTF8.GetBytes("aaa.xml"),
                Name = "aaa.xml"
            };

            var txtFile = new File
            {
                Content = Encoding.UTF8.GetBytes("bbb.txt"),
                Name = "bbb.txt"
            };

            var jsonFile = new File
            {
                Content = Encoding.UTF8.GetBytes("ccc.json"),
                Name = "ccc.json"
            };

            var dir1 = new Directory();
            dir1.AddEntry(txtFile);
            dir1.AddEntry(xmlFile);

            var dir0 = new Directory();
            dir0.AddEntry(jsonFile);
            dir0.AddEntry(dir1);

            var searcher = new FileSearcher();
            Console.WriteLine(searcher.Search(dir0, params_));
        }

        public interface IEntry
        {
            string Name { get; set; }
            int Size { get; }
            bool IsDirectory { get; }
        }

        public abstract class Entry : IEntry
        {
            public string Name { get; set; }
            public abstract int Size { get; }
            public abstract bool IsDirectory { get; }
        }

        public class File : Entry
        {
            public byte[] Content { get; set; }

            public string Extension => Name.Substring(Name.IndexOf(".") + 1);

            public override int Size => Content.Length;

            public override bool IsDirectory => false;

            public override string ToString()
            {
                return $"File{{ name='{Name}' }}";
            }
        }

        public class Directory : Entry
        {
            private List<Entry> entries = new List<Entry>();

            public override int Size => entries.Sum(entry => entry.Size);

            public override bool IsDirectory => true;

            public void AddEntry(Entry entry)
            {
                entries.Add(entry);
            }
        }

        public class SearchParams
        {
            public string Extension { get; set; }
            public int? MinSize { get; set; }
            public int? MaxSize { get; set; }
            public string Name { get; set; }
        }

        public interface IFilter
        {
            bool IsValid(SearchParams params_, File file);
        }

        public class ExtensionFilter : IFilter
        {
            public bool IsValid(SearchParams params_, File file)
            {
                if (params_.Extension == null)
                {
                    return true;
                }

                return file.Extension.Equals(params_.Extension);
            }
        }

        public class MinSizeFilter : IFilter
        {
            public bool IsValid(SearchParams params_, File file)
            {
                if (params_.MinSize == null)
                {
                    return true;
                }

                return file.Size >= params_.MinSize;
            }
        }

        public class MaxSizeFilter : IFilter
        {
            public bool IsValid(SearchParams params_, File file)
            {
                if (params_.MaxSize == null)
                {
                    return true;
                }

                return file.Size <= params_.MaxSize;
            }
        }

        public class NameFilter : IFilter
        {
            public bool IsValid(SearchParams params_, File file)
            {
                if (params_.Name == null)
                {
                    return true;
                }

                return file.Name.Equals(params_.Name);
            }
        }

        public class FileFilter
        {
            private readonly List<IFilter> filters = new List<IFilter>();

            public FileFilter()
            {
                filters.Add(new NameFilter());
                filters.Add(new MaxSizeFilter());
                filters.Add(new MinSizeFilter());
                filters.Add(new ExtensionFilter());
            }
        }

        public class FileSearcher
        {
            public string Search(Directory dir, SearchParams params_)
            {
                // Implementation not provided in the original code
                return "Search result";
            }
        }
    }




}
