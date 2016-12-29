#!/bin/bash
ruby extconf.rb
sudo make
sudo make install
sudo bundle exec rake install
ruby b.rb
