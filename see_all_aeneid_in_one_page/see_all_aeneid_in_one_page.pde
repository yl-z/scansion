import processing.pdf.*;

PFont f;
  int newcolumn = 0;
  int start = 20;
  int page = 350;
  int j = 0;
  size(2000,1100, PDF,"highlights.pdf");
  background(255);
  f = createFont("Times New Roman", 16, true);
  String[] lines = loadStrings("twelvebooks.txt");
  textFont(f,2);
  for (int i = 0 ; i < 10000 ; i++) {
    if(match(lines[i], "loc[u|o|i|a][s|m|r]*u*m*[ |,|.|;|:|?|\'|\"|q|!|)|(|]") != null) {
      fill(255,0,0);
    }
    else {fill(0,0,0,50);}
    
    if (match(lines[i], "P. VERGILI MARONIS") != null | j  == page) {
      newcolumn ++;
      j = 0;
    }
    else {j++;}
    
    text(lines[i], newcolumn * 55 - 35 , start + 3*(j));
  }
    exit();
