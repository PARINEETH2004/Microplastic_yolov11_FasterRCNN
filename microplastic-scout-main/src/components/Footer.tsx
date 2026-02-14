import { Microscope, Heart } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-border bg-card/50">
      <div className="container mx-auto px-6 py-12">
        <div className="grid md:grid-cols-3 gap-8">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Microscope className="h-5 w-5 text-primary" />
              </div>
              <span className="font-semibold text-foreground">MicroPlastic AI</span>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-powered microplastic detection and analysis platform using YOLOv11s 
              with Faster R-CNN refinement and simulated LDIR spectroscopy.
            </p>
          </div>

          {/* Technology */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground">Technology Stack</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• YOLOv11s for fast object detection</li>
              <li>• Faster R-CNN for refined accuracy</li>
              <li>• Simulated LDIR spectroscopy module</li>
              <li>• React + TypeScript frontend</li>
            </ul>
          </div>

          {/* Phase info */}
          <div className="space-y-4">
            <h4 className="font-semibold text-foreground">Development Phase</h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-primary/10 text-primary font-medium">
                  Phase-1
                </span>
                {' '}Current release
              </p>
              <p>
                Phase-2 will integrate real LDIR hardware for accurate polymer identification.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <p className="text-sm text-muted-foreground">
            © 2026 MicroPlastic Detection Platform. Academic research project.
          </p>
          <p className="text-sm text-muted-foreground flex items-center gap-1">
            Built with <Heart className="h-4 w-4 text-destructive" /> for environmental research
          </p>
        </div>
      </div>
    </footer>
  );
}
