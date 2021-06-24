#include "Exception.hpp"
#include <iostream>



Exception::Exception(std::string sum, std::string prob) //Exception constructor
{
  problem = prob;
  summary = sum;
}



void Exception::DebugPrint() //Debug method
{
  std::cerr << "**  Exception ("<<summary<<") **\n";
  std::cerr << "Problem: " << problem << "\n\n";
}
