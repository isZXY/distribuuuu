#!/bin/bash
    ssh 192.168.1.18  << remotessh
        rm -rf ~/distribuuuu/
        exit
remotessh


scp -r ~/distribuuuu root@192.168.1.18:~/distribuuuu