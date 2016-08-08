$LOAD_PATH.unshift File.expand_path('./../lib', __FILE__)
require 'tensorflow'
require 'pry'

# Requires supporting ruby files with custom matchers and macros, etc,
# in spec/support/ and its subdirectories.
Dir[File.dirname(__FILE__) + "/support/*.rb"].each {|f| require f }

RSpec.configure do |c|
  c.example_status_persistence_file_path = "examples.txt"
end
