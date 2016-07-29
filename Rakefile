require 'bundler/gem_tasks'
require 'rspec/core/rake_task'
require 'rubocop/rake_task'
require 'yard'
RSpec::Core::RakeTask.new(:spec)

YARD::Rake::YardocTask.new(:doc) do |t|
  t.files = ['lib/*.rb', 'lib/**/*.rb']
end


task :pry do |task|
  cmd = [ 'pry', "-r './lib/tensorflow.rb' "]
  run *cmd
end

def run *cmd
  sh(cmd.join(" "))
end


task :default => :spec
