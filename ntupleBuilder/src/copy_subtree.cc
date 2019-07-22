#include <string.h>

void copy_subtree(char* in_filename, char* out_filename, char* in_tree, bool odd)
{
  TFile *oldfile = new TFile(in_filename);
  TTree *oldtree = (TTree*)oldfile->Get(in_tree);
  Long64_t nentries = oldtree->GetEntries();

  if (strcmp(in_filename, out_filename) == 0)
  {
    std::cout << "input and output files have same name";
    exit(1);
  }

  std::cout << "old tree: " << in_filename << " : " << nentries << "\n";
  TFile *newfile = new TFile(out_filename, "recreate");
  TTree *newtree = oldtree->CloneTree(0);

  for (Long64_t i=0;i<nentries; i++)
  {
    oldtree->GetEntry(i);
    ULong64_t ev_number = oldtree->GetLeaf("event")->GetValue();
    if (ev_number%2 == odd) newtree->Fill();
  }

  std::cout << "new tree: " << out_filename << " : " << newtree->GetEntries() << "\n";
  newtree->AutoSave();
  delete oldfile;
  delete newfile;
  exit(0);
}
