import { Microscope, Github, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 glass-panel border-b">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Microscope className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h1 className="font-semibold text-foreground">MicroPlastic AI</h1>
            <p className="text-xs text-muted-foreground">YOLOv11 + LDIR Analysis</p>
          </div>
        </div>

        <nav className="flex items-center gap-2">
          <Button variant="ghost" size="sm" className="gap-2">
            <FileText className="h-4 w-4" />
            <span className="hidden sm:inline">Documentation</span>
          </Button>

        </nav>
      </div>
    </header>
  );
}
