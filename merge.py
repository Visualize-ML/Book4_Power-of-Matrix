#!/bin/env python3

import os
import datetime

# pip3 install pypdf 
from pypdf import PdfWriter, PdfReader, PageRange


mfname= "book4_power_of_matrix"
def main():

    pdfwrtr = PdfWriter()
    t_pages = 0
    net_start = False
    for r, dirs, files in os.walk("."):
        del dirs[:]

        for fitem in files:
            fname, fext = os.path.splitext(fitem) 

            if fext.lower() != ".pdf":
                continue 
            if not fitem.startswith("Book"):
                continue
            
            pdfrd = PdfReader(fitem, 'rb')

            meta = pdfrd.metadata 
            
            fname = fname.replace("Book4_", "")
            fname = fname.split("__")[0]
            fname = fname.replace("_", " ")
            
            pdfwrtr.append( pdfrd, fname  )
            pdfwrtr.add_metadata(meta)
    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y.%m.%d_%H_%M_%S")
    pdfwrtr.write("{mfname}.{nw}.pdf".format(mfname=mfname, nw=now_str))
    pdfwrtr.close()


if __name__ == '__main__':
    main()
